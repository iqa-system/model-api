from __future__ import annotations

# 这个文件负责 "模型注册表".
# 如果把 model_adapters.py 看成 "单个模型怎么包装",
# 那么这个文件就是 "多个模型怎么统一装配和管理".
# 服务启动时会读取 JSON 配置, 创建 adapter, 加载权重, 并记录默认模型是谁.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from model_adapters import BaseIqaModelAdapter, ModelDescriptor, ModelPrediction, create_adapter


# SERVICE_ROOT 指向服务代码根目录.
# 解析相对权重路径时会优先参考这里.
SERVICE_ROOT = Path(__file__).resolve().parent


class ModelConfig(BaseModel):
    # 这是配置文件中单个模型项的结构定义.
    # Pydantic 会负责做基础字段校验.
    id: str
    display_name: str
    description: str
    adapter: str
    weights_path: str
    enabled: bool = True
    adapter_options: dict[str, Any] = Field(default_factory=dict)


class RegistryConfig(BaseModel):
    # 整个模型注册表配置文件的结构定义.
    default_model_id: str
    models: list[ModelConfig]


@dataclass
class ModelRegistry:
    # 运行期注册表对象.
    # default_model_id 表示默认模型.
    # adapters 是 "model_id -> adapter 实例" 的映射.
    # model_order 用来保留配置文件里的展示顺序.
    # config_path 只是记录本次实际加载的是哪份配置文件, 便于排查问题.
    default_model_id: str
    adapters: dict[str, BaseIqaModelAdapter]
    model_order: list[str]
    config_path: Path

    @classmethod
    def from_path(cls, config_path: str | Path, device: torch.device) -> "ModelRegistry":
        # 这是注册表最重要的工厂方法.
        # 它完成的事情包括:
        # 1. 读取 JSON.
        # 2. 校验配置是否合法.
        # 3. 创建每个 adapter.
        # 4. 在启动阶段预加载所有启用模型.
        resolved_config_path = Path(config_path).resolve()
        if not resolved_config_path.exists():
            raise RuntimeError(f"模型注册表不存在: {resolved_config_path}")

        with resolved_config_path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)

        # Pydantic V2 推荐使用 model_validate.
        # 这里保留对旧版本的兼容, 这样不同环境下都能工作.
        if hasattr(RegistryConfig, "model_validate"):
            config = RegistryConfig.model_validate(raw_config)
        else:
            config = RegistryConfig.parse_obj(raw_config)
        if not config.models:
            raise RuntimeError("模型注册表中没有任何模型配置")

        all_ids = [model.id for model in config.models]
        # model_id 必须唯一.
        # 否则外部传入一个 model_id 时, 服务无法确定该路由到哪个模型.
        duplicate_ids = sorted({model_id for model_id in all_ids if all_ids.count(model_id) > 1})
        if duplicate_ids:
            duplicate_str = ", ".join(duplicate_ids)
            raise RuntimeError(f"模型注册表存在重复的 model_id: {duplicate_str}")

        # enabled=false 的模型会被忽略, 不会参与启动加载, 也不会出现在 /models 中.
        enabled_models = [model for model in config.models if model.enabled]
        if not enabled_models:
            raise RuntimeError("模型注册表中没有启用的模型")

        enabled_ids = {model.id for model in enabled_models}
        # 默认模型必须存在, 而且必须是启用状态.
        if config.default_model_id not in enabled_ids:
            raise RuntimeError(f"default_model_id 未启用或不存在: {config.default_model_id}")

        adapters: dict[str, BaseIqaModelAdapter] = {}
        model_order: list[str] = []
        for model in enabled_models:
            # 先解析权重路径.
            # 这一步支持绝对路径, 服务根目录相对路径, 以及配置文件相对路径.
            weights_path = _resolve_weights_path(resolved_config_path, model.weights_path)
            if not weights_path.exists():
                raise RuntimeError(f"模型权重不存在: {weights_path}")

            # 再根据 adapter 名称创建对应 adapter.
            adapter = create_adapter(
                adapter_name=model.adapter,
                model_id=model.id,
                display_name=model.display_name,
                description=model.description,
                weights_path=weights_path,
                adapter_options=model.adapter_options,
            )
            try:
                # 这里是启动阶段真正加载模型权重的地方.
                # 这样请求进来时就不需要再做首次加载等待.
                adapter.load(device)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"加载模型失败: {model.id}") from exc

            adapters[model.id] = adapter
            model_order.append(model.id)

        return cls(
            default_model_id=config.default_model_id,
            adapters=adapters,
            model_order=model_order,
            config_path=resolved_config_path,
        )

    def resolve_model_id(self, requested_model_id: str | None) -> str:
        # 统一处理 "调用方有没有传 model_id" 这件事.
        # 没传就回退到默认模型.
        # 传了但不存在, 就抛出明确错误.
        model_id = requested_model_id or self.default_model_id
        if model_id not in self.adapters:
            available = ", ".join(self.model_order)
            raise ValueError(f"未知模型: {model_id}。当前可用模型: {available}")
        return model_id

    def get_adapter(self, requested_model_id: str | None = None) -> BaseIqaModelAdapter:
        # 如果上层只想拿 adapter 自己, 可以调用这个方法.
        model_id = self.resolve_model_id(requested_model_id)
        return self.adapters[model_id]

    def predict(self, requested_model_id: str | None, image) -> tuple[str, ModelPrediction]:
        # 这是注册表暴露给上层最常用的方法.
        # 返回值里同时带上最终解析后的 model_id, 这样接口层不用再自己猜到底用了哪个模型.
        model_id = self.resolve_model_id(requested_model_id)
        return model_id, self.adapters[model_id].predict(image)

    def list_models(self) -> list[ModelDescriptor]:
        # 按配置顺序输出模型列表, 并标记哪个是默认模型.
        return [
            self.adapters[model_id].descriptor(is_default=(model_id == self.default_model_id))
            for model_id in self.model_order
        ]


def _resolve_weights_path(config_path: Path, weights_path: str) -> Path:
    # 权重路径解析规则如下:
    # 1. 如果本身就是绝对路径, 直接使用.
    # 2. 如果相对于服务根目录存在, 优先使用.
    # 3. 否则按相对于配置文件所在目录解析.
    # 这样兼顾了本地开发和部署环境两种使用方式.
    raw_path = Path(weights_path)
    if raw_path.is_absolute():
        return raw_path

    service_relative = (SERVICE_ROOT / raw_path).resolve()
    if service_relative.exists():
        return service_relative

    return (config_path.parent / raw_path).resolve()
