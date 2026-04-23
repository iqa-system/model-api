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


def _image_to_batch_tensor(image: Image.Image, device: torch.device):
    # 这个辅助函数负责把 PIL 图片转换成模型可直接消费的 4 维 Tensor.
    # 大多数 PyTorch 图像模型都要求输入形状是 [N, C, H, W].
    # 这里统一补上 batch 维度, 避免每个 adapter 都重复写一遍同样的转换逻辑.
    from torchvision.transforms import ToTensor

    return ToTensor()(image).unsqueeze(0).to(device)


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

        # 这里复用统一的图片转 Tensor 辅助函数.
        # 这样不同 adapter 的输入预处理风格会更一致.
        img_tensor = _image_to_batch_tensor(image, self.device)
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


class PyiqaMetricAdapter(BaseIqaModelAdapter):
    # 这个 adapter 用来对接 pyiqa 里的通用图像质量模型.
    # 当前我们主要拿它接 MUSIQ 和 TOPIQ-NR.
    # 之所以做成通用 adapter, 是因为后续如果还要接别的 pyiqa 指标,
    # 通常只需要在 JSON 里换一个 metric_name 和权重文件即可.
    def __init__(
        self,
        model_id: str,
        display_name: str,
        description: str,
        weights_path: Path,
        adapter_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id, display_name, description, weights_path, adapter_options)
        self.device: torch.device | None = None
        self.model = None

    @property
    def capabilities(self) -> list[str]:
        # pyiqa 这类 NR 评分模型默认只返回质量分数.
        # 如果未来某个包装模型还想暴露额外字段, 可以在配置里覆盖 capabilities.
        configured_capabilities = self.adapter_options.get("capabilities")
        if isinstance(configured_capabilities, list) and configured_capabilities:
            return [str(item) for item in configured_capabilities]
        return ["quality_score"]

    def load(self, device: torch.device) -> None:
        # pyiqa 依赖体积较大, 这里继续采用延迟导入.
        # 只有真的加载到这个 adapter 时, 才去导入对应库.
        try:
            import pyiqa
        except ModuleNotFoundError as exc:
            raise RuntimeError("当前环境未安装 pyiqa, 无法加载该模型") from exc

        metric_name = self._get_metric_name()
        load_strategy = str(self.adapter_options.get("load_strategy", "constructor_path")).strip()
        metric_options = self._build_metric_options(load_strategy)

        self.device = device
        self.model = pyiqa.create_metric(metric_name, device=device, **metric_options)

        # 有些 pyiqa 模型支持在构造函数里直接接收 pretrained_model_path.
        # 也有些模型更适合先建网络, 再手动 load_weights.
        # 这里通过 load_strategy 兼容两种方式, 但默认优先走 constructor_path,
        # 因为它通常能保留 pyiqa 原作者在各个模型内部定义的加载细节.
        if load_strategy == "manual_load_weights":
            weight_keys = "params"
            if "weight_keys" in self.adapter_options:
                weight_keys = self.adapter_options["weight_keys"]
            self.model.load_weights(str(self.weights_path), weight_keys=weight_keys)
        elif load_strategy != "constructor_path":
            raise ValueError(
                "pyiqa adapter 的 load_strategy 仅支持 constructor_path 或 manual_load_weights"
            )

        self.model.eval()

    def predict(self, image: Image.Image) -> ModelPrediction:
        if self.model is None or self.device is None:
            raise RuntimeError(f"模型未加载: {self.model_id}")

        img_tensor = _image_to_batch_tensor(image, self.device)
        with torch.no_grad():
            raw_score = self.model(img_tensor)

        # pyiqa 大多数指标返回的是单元素 Tensor.
        # 为了让 API 响应稳定, 这里统一压成 Python float.
        quality_score = self._to_float_score(raw_score)
        return ModelPrediction(
            quality_score=quality_score,
            extra_outputs={},
        )

    def _get_metric_name(self) -> str:
        # metric_name 是 pyiqa.create_metric 的核心入参.
        # 比如 "musiq" 或 "topiq_nr".
        metric_name = str(self.adapter_options.get("metric_name", "")).strip()
        if not metric_name:
            raise ValueError("pyiqa adapter 缺少 adapter_options.metric_name 配置")
        return metric_name

    def _build_metric_options(self, load_strategy: str) -> dict[str, Any]:
        # metric_options 会原样透传给 pyiqa.create_metric.
        # 这样我们既能保留默认配置, 也能在 JSON 里细调某些模型参数.
        raw_metric_options = self.adapter_options.get("metric_options", {})
        if raw_metric_options is None:
            raw_metric_options = {}
        if not isinstance(raw_metric_options, dict):
            raise ValueError("pyiqa adapter 的 adapter_options.metric_options 必须是对象")

        metric_options = dict(raw_metric_options)
        if load_strategy == "constructor_path":
            # 这是这次改造最关键的一步.
            # 我们显式把项目内的本地权重路径塞给 pyiqa,
            # 从而阻止它退回到用户目录缓存或远程下载逻辑.
            metric_options["pretrained_model_path"] = str(self.weights_path)
        return metric_options

    @staticmethod
    def _to_float_score(raw_score: Any) -> float:
        # 某些模型可能返回 tuple 或 list.
        # 对质量评估服务来说, 我们只取第一个主分数作为 quality_score.
        if isinstance(raw_score, (tuple, list)):
            if not raw_score:
                raise RuntimeError("pyiqa 模型返回了空结果")
            raw_score = raw_score[0]

        if hasattr(raw_score, "item"):
            return float(raw_score.item())
        return float(raw_score)


ADAPTER_TYPES: dict[str, type[BaseIqaModelAdapter]] = {
    # 这是一个简单的 adapter 注册表.
    # key 是配置文件里写的 adapter 名称.
    # value 是真正的 Python 类.
    "liqe": LiqeAdapter,
    "pyiqa": PyiqaMetricAdapter,
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
