from __future__ import annotations

# 这个文件是 FastAPI 服务入口.
# 主要职责有 3 件事:
# 1. 在服务启动时读取模型注册表, 并预加载所有启用模型.
# 2. 定义对外 HTTP 接口, 比如 /models, /predict, /predict/batch.
# 3. 把内部模型推理结果整理成稳定的 API 响应结构.

from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
from typing import Any, NoReturn

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, ImageFile
from pydantic import BaseModel, Field

from model_registry import ModelRegistry


# BASE_DIR 指向当前文件所在目录.
# 这样无论服务从哪里启动, 都能稳定找到 config/model_registry.json.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_REGISTRY_PATH = BASE_DIR / "config" / "model_registry.json"

# 允许 Pillow 读取部分被截断的图片.
# 这对线上服务比较有用, 因为用户上传的图片不一定总是完全规范.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 推理前允许的最长边, 默认 2048 像素.
# 注意 2 MB 的 JPEG 并不代表解码后很小, 例如高压缩图片可能有几千万像素.
# LIQE 会把图像展开成很多 224x224 patch, 高分辨率图片会迅速放大内存和显存占用.
# 因此这里默认按最长边等比例缩放, 优先保证服务稳定可用.
MAX_INFERENCE_LONG_SIDE = int(os.getenv("IQA_MAX_INFERENCE_LONG_SIDE", "2048"))

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    # 兼容旧版 Pillow. 旧版 Pillow 还没有 Image.Resampling 枚举.
    RESAMPLE_LANCZOS = Image.LANCZOS


class DistortionProb(BaseModel):
    # 这个 Pydantic 模型描述单个失真候选项.
    # index 是失真类别编码.
    # prob 是该失真类别对应的概率.
    index: int = Field(description="失真类型编码")
    prob: float = Field(description="该失真类型对应的概率")


class ScoreRange(BaseModel):
    # 统一描述模型原始分数区间.
    # 当前服务约定所有模型都满足 "分数越大越好", 因此这里只返回最小值和最大值.
    min_score: float = Field(description="模型原始分数最小值")
    max_score: float = Field(description="模型原始分数最大值")


class PredictResult(BaseModel):
    # 这是单张图片推理接口的统一响应结构.
    # 不同模型至少要返回 model_id 和 quality_score.
    # scene, distortion, distortions 是 LIQE 这类模型的附加能力, 所以允许为空.
    # extra_outputs 预留给未来其它模型使用, 比如热力图分数, 置信度向量等.
    file_name: str | None = Field(default=None, description="上传文件名. 批量接口中通常会返回")
    model_id: str = Field(description="实际执行推理的模型 ID")
    quality_score: float = Field(description="模型输出的质量分数")
    score_range: ScoreRange = Field(description="该模型原始分数的取值区间")
    extra_outputs: dict[str, Any] = Field(
        default_factory=dict,
        description="模型的通用扩展输出. 当前 pyiqa 模型通常为空对象",
    )
    scene: int | None = Field(default=None, description="场景类别编码. 仅 LIQE 等支持该能力的模型返回")
    distortion: int | None = Field(
        default=None,
        description="主失真类别编码. 仅 LIQE 等支持该能力的模型返回",
    )
    distortions: list[DistortionProb] | None = Field(
        default=None,
        description="失真概率列表. 仅 LIQE 等支持该能力的模型返回",
    )


class BatchPredictResponse(BaseModel):
    # 批量推理接口的响应很简单, 就是多个单图结果的列表.
    results: list[PredictResult] = Field(description="批量推理结果列表")


class ModelInfo(BaseModel):
    # /models 接口会返回这个结构, 供 SpringBoot 或前端生成模型下拉框.
    model_id: str = Field(description="模型唯一 ID. 调用 /predict 时通过 model_id 选择模型")
    display_name: str = Field(description="用于前端展示的模型名称")
    description: str = Field(description="模型特点说明, 包括适用场景, 优势或性能概述")
    capabilities: list[str] = Field(description="当前模型可返回的能力字段列表")
    score_range: ScoreRange = Field(description="该模型原始分数的取值区间")
    is_default: bool = Field(description="该模型是否为默认模型")


class ModelsResponse(BaseModel):
    # /models 的顶层返回结构.
    # default_model_id 用来告诉调用方 "不传 model_id 时会走哪个模型".
    default_model_id: str = Field(description="默认模型 ID. 当 /predict 未传 model_id 时会使用它")
    models: list[ModelInfo] = Field(description="当前已启用的模型列表")


def _set_deterministic(seed: int = 20200626) -> None:
    # 固定随机种子, 让同一张图在相同环境下更容易得到可复现结果.
    # 这在排查 "为什么这次和上次结果不一样" 时很有帮助.
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_device() -> torch.device:
    # 优先使用 GPU, 没有 GPU 时退回 CPU.
    # torch.device 是 PyTorch 用来表示运行设备的对象.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


async def _read_upload_image(file: UploadFile) -> Image.Image:
    # FastAPI 的 UploadFile 是异步文件对象.
    # 这里把上传内容读入内存, 然后交给 Pillow 解码.
    # convert("RGB") 的目的是统一颜色通道格式, 避免后续模型处理时出现模式差异.
    try:
        content = await file.read()
        # 文件体积上限由 Java 后端 SpringBoot 统一控制.
        # 模型服务这里只负责解码图片, 并根据像素尺寸做推理前缩放保护.
        image = Image.open(BytesIO(content)).convert("RGB")
        return _resize_image_for_inference(image, file.filename or "unknown")
    except Exception as exc:  # noqa: BLE001
        # 这里统一把各种底层图片解码错误包装成 400.
        # 对调用方来说, 只需要知道图片不可读即可, 不需要看到过于底层的异常细节.
        raise HTTPException(status_code=400, detail=f"图片读取失败: {file.filename}") from exc


def _resize_image_for_inference(image: Image.Image, file_name: str) -> Image.Image:
    # 这里根据像素尺寸而不是文件体积判断是否需要缩放.
    # JPEG/PNG 是压缩格式, 2 MB 的文件也可能解码成几千万个像素.
    # 模型真正消耗的是解码后的像素张量和中间特征, 所以必须看 width 和 height.
    width, height = image.size
    long_side = max(width, height)

    processed_image = image

    if 0 < MAX_INFERENCE_LONG_SIDE < long_side:
        # thumbnail 会保持原始宽高比, 并且只会缩小不会放大.
        # 这里先 copy, 避免修改原始 PIL 对象带来隐式副作用.
        processed_image = image.copy()
        processed_image.thumbnail(
            (MAX_INFERENCE_LONG_SIDE, MAX_INFERENCE_LONG_SIDE),
            RESAMPLE_LANCZOS,
        )
        resized_width, resized_height = processed_image.size
        print(
            "图片尺寸较大, 已在推理前等比例缩放: "
            f"{file_name}, original={width}x{height}, resized={resized_width}x{resized_height}"
        )

    final_width, final_height = processed_image.size
    print(
        "图片预处理完成: "
        f"{file_name}, original={width}x{height}, final={final_width}x{final_height}"
    )
    return processed_image


def _raise_inference_error(exc: Exception) -> NoReturn:
    # 这里把模型底层异常转换成稳定的 JSON 错误响应.
    # 如果不捕获, Java 后端可能拿到非 JSON 响应, 进而出现 DTO 反序列化失败.
    # 如果是 CUDA OOM, empty_cache 可以尽量释放 PyTorch 缓存, 方便后续请求继续尝试.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"模型推理失败: {type(exc).__name__}: {exc}")
    raise HTTPException(
        status_code=500,
        detail=(
            "模型推理失败, 可能是图片分辨率过大, 内存不足或显存不足. "
            "请查看模型服务日志确认具体异常."
        ),
    ) from exc


def _get_registry(request: Request) -> ModelRegistry:
    # FastAPI 的 app.state 可以存放整个应用生命周期共享的数据.
    # 我们把模型注册表放进去, 这样每个请求都能复用已加载好的模型实例.
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        # 正常情况下不会走到这里.
        # 如果出现了, 一般说明服务启动流程没有正确完成.
        raise RuntimeError("模型注册表尚未初始化")
    return registry


def _build_predict_result(
    file_name: str | None,
    model_id: str,
    prediction,
    score_range: ScoreRange,
) -> PredictResult:
    # adapter 返回的是内部通用结构 ModelPrediction.
    # 这里负责把内部结构转换成对外 API 响应结构 PredictResult.
    # 这样可以把 "内部实现" 和 "对外协议" 分开管理.
    distortion_items = None
    if prediction.distortions is not None:
        # 内部失真列表是普通字典列表.
        # 这里转成 DistortionProb, 方便 FastAPI 自动生成更清晰的 OpenAPI 文档.
        distortion_items = [
            DistortionProb(index=item["index"], prob=item["prob"])
            for item in prediction.distortions
        ]

    return PredictResult(
        file_name=file_name,
        model_id=model_id,
        quality_score=prediction.quality_score,
        score_range=score_range,
        extra_outputs=dict(prediction.extra_outputs),
        scene=prediction.scene,
        distortion=prediction.distortion,
        distortions=distortion_items,
    )


def _build_model_info(descriptor) -> ModelInfo:
    # registry.list_models() 返回的是内部 dataclass ModelDescriptor.
    # 其中的 score_range 也是内部 dataclass ScoreRange, 不能直接原样塞给
    # API 层的 Pydantic ModelInfo, 否则 Pydantic 会把它当成“未知对象类型”并报校验错误.
    # 这里显式做一层转换, 统一把内部描述对象映射成对外 API 契约对象.
    return ModelInfo(
        model_id=descriptor.model_id,
        display_name=descriptor.display_name,
        description=descriptor.description,
        capabilities=list(descriptor.capabilities),
        score_range=ScoreRange(
            min_score=descriptor.score_range.min_score,
            max_score=descriptor.score_range.max_score,
        ),
        is_default=descriptor.is_default,
    )


def _create_lifespan(model_registry_path: Path):
    # FastAPI 的 lifespan 是应用启动和关闭时运行的钩子.
    # 我们用它来做 "服务启动时一次性加载所有模型" 这件事.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("服务启动中: 加载模型注册表...")
        # 先固定随机性, 再选择设备, 最后加载注册表.
        # 这个顺序比较自然, 也方便排查启动问题.
        _set_deterministic()
        device = _get_device()
        registry = ModelRegistry.from_path(model_registry_path, device)
        # 把设备和注册表挂到 app.state 上.
        # 后续每个请求都从这里取, 而不是重复创建模型对象.
        app.state.device = device
        app.state.model_registry = registry
        print(
            f"已加载 {len(registry.adapters)} 个模型, "
            f"默认模型: {registry.default_model_id}, "
            f"设备: {device}"
        )
        yield

    return lifespan


def create_app(model_registry_path: Path | None = None) -> FastAPI:
    # create_app 的好处是既能给线上运行用, 也方便测试时传入临时配置文件.
    registry_path = Path(model_registry_path or DEFAULT_MODEL_REGISTRY_PATH).resolve()
    app = FastAPI(title="IQA Model API", version="2.0", lifespan=_create_lifespan(registry_path))

    @app.get(
        "/health",
        tags=["system"],
        summary="健康检查",
        description="返回服务状态, 默认模型 ID, 以及当前已成功加载的模型列表.",
    )
    def health_check(request: Request) -> dict[str, Any]:
        # 健康检查通常给运维系统, 反向代理, 或上层服务使用.
        # 除了告诉调用方服务活着, 也顺便返回当前加载了哪些模型.
        registry = getattr(request.app.state, "model_registry", None)
        loaded_models = registry.model_order if registry else []
        return {
            "status": "ok",
            "model_loaded": bool(loaded_models),
            "default_model_id": registry.default_model_id if registry else None,
            "loaded_models": loaded_models,
        }

    @app.get(
        "/models",
        response_model=ModelsResponse,
        tags=["models"],
        summary="获取可用模型列表",
        description=(
            "返回当前所有启用模型的 ID, 展示名, 特点说明和能力字段. "
            "前端可以用这个接口渲染模型下拉框, 也可以据此决定是否展示 scene 或 distortion."
        ),
    )
    def list_models(request: Request) -> ModelsResponse:
        # 这个接口专门给外部系统发现模型用.
        # 前端可以据此渲染模型下拉框.
        # SpringBoot 也可以据此决定是否做一层转发或缓存.
        registry = _get_registry(request)
        return ModelsResponse(
            default_model_id=registry.default_model_id,
            models=[_build_model_info(descriptor) for descriptor in registry.list_models()],
        )

    @app.post(
        "/predict",
        response_model=PredictResult,
        tags=["inference"],
        summary="单张图片推理",
        description=(
            "上传一张图片并返回质量评估结果. "
            "如果未传 model_id, 服务会自动使用 /models 返回的 default_model_id."
        ),
    )
    async def predict(
        request: Request,
        file: UploadFile = File(..., description="待评估图片文件"),
        model_id: str | None = Form(
            None,
            description="可选模型 ID. 当前可用值可通过 /models 查询, 未传时使用默认模型",
        ),
    ) -> PredictResult:
        # 单图推理的流程是:
        # 1. 校验上传文件.
        # 2. 从 app.state 中拿到注册表.
        # 3. 读取图片.
        # 4. 根据 model_id 选择模型并推理.
        # 5. 统一封装响应.
        if not file.filename:
            raise HTTPException(status_code=400, detail="未提供图片文件")

        registry = _get_registry(request)
        img = await _read_upload_image(file)
        try:
            # registry.predict 会自动处理两种情况:
            # 1. model_id 为 None 时, 使用默认模型.
            # 2. model_id 有值时, 校验该模型是否存在并已启用.
            resolved_model_id, prediction = registry.predict(model_id, img)
        except ValueError as exc:
            # 这里的 ValueError 一般代表调用方传了未知模型 ID.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            # 模型推理阶段可能出现 CUDA OOM, CPU 内存不足, 或第三方模型内部错误.
            # 统一包装成 JSON 响应, 方便 Java 后端识别和展示.
            _raise_inference_error(exc)

        return _build_predict_result(
            file.filename,
            resolved_model_id,
            prediction,
            ScoreRange(**registry.get_adapter(resolved_model_id).score_range.__dict__),
        )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictResponse,
        tags=["inference"],
        summary="多张图片批量推理",
        description=(
            "一次上传多张图片并使用同一个模型批量评估. "
            "如果未传 model_id, 服务会自动使用默认模型."
        ),
    )
    async def predict_batch(
        request: Request,
        files: list[UploadFile] = File(..., description="待批量评估的图片文件列表"),
        model_id: str | None = Form(
            None,
            description="可选模型 ID. 当前可用值可通过 /models 查询, 未传时使用默认模型",
        ),
    ) -> BatchPredictResponse:
        # 批量推理和单图推理思路一致.
        # 差别是这里先把 model_id 解析一次, 然后循环处理每一张图.
        # 这样可以避免在循环里重复做同一套模型选择校验.
        if not files:
            raise HTTPException(status_code=400, detail="未提供图片文件列表")

        registry = _get_registry(request)
        results: list[PredictResult] = []
        try:
            resolved_model_id = registry.resolve_model_id(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for file in files:
            # 每张图片都单独读取和推理.
            # 这样即使以后要做更细粒度的错误处理, 这里也比较容易扩展.
            img = await _read_upload_image(file)
            try:
                _, prediction = registry.predict(resolved_model_id, img)
            except Exception as exc:  # noqa: BLE001
                _raise_inference_error(exc)
            results.append(
                _build_predict_result(
                    file.filename,
                    resolved_model_id,
                    prediction,
                    ScoreRange(**registry.get_adapter(resolved_model_id).score_range.__dict__),
                )
            )

        return BatchPredictResponse(results=results)

    return app


app = create_app()
# 上面这行会在 import api_server 时直接创建 FastAPI 应用对象.
# Uvicorn 在启动时会读取这个 app 变量.


if __name__ == "__main__":
    import uvicorn

    # 直接运行 python api_server.py 时, 会启动 Uvicorn 服务.
    # Debian 服务器部署时必须监听 0.0.0.0, 才能接受来自外部机器的请求.
    # 如果只监听 localhost 或 127.0.0.1, 服务虽然在本机运行成功, 但外部访问会出现 ECONNREFUSED.
    # 这里同时支持通过环境变量覆盖默认值, 方便你在不同服务器或端口上演示.
    host = os.getenv("IQA_API_HOST", "0.0.0.0")
    port = int(os.getenv("IQA_API_PORT", "8000"))
    uvicorn.run("api_server:app", host=host, port=port, reload=False)
