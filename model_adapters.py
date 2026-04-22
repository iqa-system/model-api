from __future__ import annotations

# 这个文件定义 "模型适配层".
# 适配层的核心作用是把不同模型包装成统一接口.
# 这样上层服务不需要知道某个模型到底叫 LIQE, MUSIQ, 还是别的名字.
# 只要模型实现了同一套 load 和 predict 方法, 注册表就能统一管理它们.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image


@dataclass
class ModelPrediction:
    # 这是模型内部统一输出结构.
    # 不管底层是什么模型, 最终都先转成这个结构, 再交给 API 层格式化.
    quality_score: float
    extra_outputs: dict[str, Any] = field(default_factory=dict)
    scene: int | None = None
    distortion: int | None = None
    distortions: list[dict[str, float]] | None = None


@dataclass
class ModelDescriptor:
    # 这是给 /models 接口准备的描述信息.
    # 它不负责推理, 只负责描述 "这个模型是谁, 有什么能力".
    model_id: str
    display_name: str
    description: str
    capabilities: list[str]
    is_default: bool


class BaseIqaModelAdapter(ABC):
    # BaseIqaModelAdapter 是所有模型 adapter 的共同父类.
    # 这里定义了一个最小契约:
    # 1. load(device) 负责加载模型.
    # 2. predict(image) 负责执行推理.
    # 3. capabilities 告诉外部这个模型支持哪些输出字段.
    def __init__(
        self,
        model_id: str,
        display_name: str,
        description: str,
        weights_path: Path,
        adapter_options: dict[str, Any] | None = None,
    ) -> None:
        # 这些字段都是所有模型都需要的基础元数据.
        # adapter_options 预留给未来更复杂模型使用, 比如阈值, 归一化参数, 特殊开关等.
        self.model_id = model_id
        self.display_name = display_name
        self.description = description
        self.weights_path = weights_path
        self.adapter_options = adapter_options or {}

    @property
    @abstractmethod
    def capabilities(self) -> list[str]:
        # 用属性而不是普通方法, 是因为能力列表本质上更像模型的元信息.
        raise NotImplementedError

    def descriptor(self, is_default: bool) -> ModelDescriptor:
        # 把 adapter 自身的元数据打包成可直接返回给上层的描述对象.
        return ModelDescriptor(
            model_id=self.model_id,
            display_name=self.display_name,
            description=self.description,
            capabilities=self.capabilities,
            is_default=is_default,
        )

    @abstractmethod
    def load(self, device: torch.device) -> None:
        # 约定每个 adapter 必须显式实现加载逻辑.
        # 这样注册表在启动时就能统一调用.
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Image.Image) -> ModelPrediction:
        # 约定每个 adapter 必须把结果统一转换成 ModelPrediction.
        raise NotImplementedError


class LiqeAdapter(BaseIqaModelAdapter):
    # 这是当前已经落地的具体 adapter, 负责对接 LIQE 模型.
    # 以后新增其它模型时, 一般也是新增一个类似的类.
    def __init__(
        self,
        model_id: str,
        display_name: str,
        description: str,
        weights_path: Path,
        adapter_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id, display_name, description, weights_path, adapter_options)
        # device 和 model 会在 load 时真正赋值.
        # 这样创建 adapter 对象本身不会立刻触发模型加载.
        self.device: torch.device | None = None
        self.model = None

    @property
    def capabilities(self) -> list[str]:
        # LIQE 当前支持质量分数, 场景, 失真主类, 失真概率列表.
        return ["quality_score", "scene", "distortion", "distortions"]

    def load(self, device: torch.device) -> None:
        # 这里延迟导入 utils.LIQE, 有两个好处:
        # 1. 测试时更容易 mock.
        # 2. 只有真正加载 LIQE 时才会触发对应依赖检查.
        from utils.LIQE import LIQE

        self.device = device
        self.model = LIQE(str(self.weights_path), device)
        # eval 模式表示当前是推理而不是训练.
        # 这样 Dropout, BatchNorm 等层会使用推理行为.
        self.model.eval()

    def predict(self, image: Image.Image) -> ModelPrediction:
        if self.model is None or self.device is None:
            raise RuntimeError(f"模型未加载: {self.model_id}")

        # ToTensor 会把 PIL 图片转成 PyTorch Tensor.
        # unsqueeze(0) 是补一个 batch 维度, 因为模型通常期望输入形状为 [N, C, H, W].
        from torchvision.transforms import ToTensor

        img_tensor = ToTensor()(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # no_grad 表示这里只做推理, 不需要记录梯度.
            # 这样可以节省显存和计算开销.
            quality, scene, distortion, distortions = self.model(img_tensor)

        # LIQE 返回的 distortions 是内部字典结构.
        # 这里统一把值转成基础 Python 类型, 避免后续 JSON 序列化时出现类型兼容问题.
        distortion_items = [
            {"index": int(item["index"]), "prob": float(item["prob"])}
            for item in distortions[0]
        ]
        return ModelPrediction(
            quality_score=float(quality.item()),
            scene=int(scene.item()),
            distortion=int(distortion.item()),
            distortions=distortion_items,
            extra_outputs={},
        )


ADAPTER_TYPES: dict[str, type[BaseIqaModelAdapter]] = {
    # 这是一个简单的 adapter 注册表.
    # key 是配置文件里写的 adapter 名称.
    # value 是真正的 Python 类.
    "liqe": LiqeAdapter,
}


def create_adapter(
    adapter_name: str,
    model_id: str,
    display_name: str,
    description: str,
    weights_path: Path,
    adapter_options: dict[str, Any] | None = None,
) -> BaseIqaModelAdapter:
    # 根据配置文件里的 adapter_name 动态创建具体 adapter.
    # 这样新增模型时, 上层代码一般不需要修改, 只需要:
    # 1. 新增 adapter 类.
    # 2. 把它注册到 ADAPTER_TYPES.
    # 3. 在 JSON 配置里引用它.
    adapter_cls = ADAPTER_TYPES.get(adapter_name)
    if adapter_cls is None:
        supported = ", ".join(sorted(ADAPTER_TYPES))
        raise ValueError(f"不支持的模型适配器: {adapter_name}。可选值: {supported}")
    return adapter_cls(
        model_id=model_id,
        display_name=display_name,
        description=description,
        weights_path=weights_path,
        adapter_options=adapter_options,
    )
