import os
from io import BytesIO
from typing import List
from contextlib import asynccontextmanager

import torch
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageFile
from torchvision.transforms import ToTensor

from utils.LIQE import LIQE

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许读取被截断的图片

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")  # 批量处理允许的扩展名


class PredictResult(BaseModel):
    # 单张图片推理的响应结构
    file_name: str | None = None
    quality_score: float
    scene: int
    distortion: int


class BatchPredictResponse(BaseModel):
    # 批量推理的响应结构
    results: List[PredictResult]


class ProcessFolderRequest(BaseModel):
    # 批处理请求体
    input_path: str
    output_path: str


class ProcessFolderResult(BaseModel):
    # 批处理结果摘要
    input_path: str
    output_path: str
    total_images: int
    processed_images: int
    skipped_images: int


def _set_deterministic(seed: int = 20200626) -> None:
    # 固定随机性,方便复现
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_device() -> torch.device:
    # 自动选择 GPU 或 CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(ckpt_path: str, device: torch.device) -> LIQE:
    # 初始化模型并切到 eval 模式
    model = LIQE(ckpt_path, device)
    model.eval()
    return model


def _predict_image(model: LIQE, device: torch.device, img: Image.Image) -> PredictResult:
    # 输入必须是 RGB 格式,调用方需保证
    # ToTensor 会将 PIL.Image 转为 [0,1] 的 float Tensor
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        # 模型输出: 质量分数,场景编码,失真编码
        q, s, d = model(img_tensor)
    return PredictResult(
        quality_score=float(q.item()),
        scene=int(s.item()),
        distortion=int(d.item()),
    )


async def _read_upload_image(file: UploadFile) -> Image.Image:
    try:
        # 读取上传文件内容,避免磁盘落地
        content = await file.read()
        img = Image.open(BytesIO(content)).convert("RGB")
        return img
    except Exception as exc:  # noqa: BLE001
        # 上传文件损坏或格式异常
        raise HTTPException(status_code=400, detail=f"图片读取失败: {file.filename}") from exc


def _save_results(results: List[dict], output_path: str) -> None:
    # 输出路径不存在时自动创建
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 统一用 DataFrame 便于写出多种格式
    df = pd.DataFrame(results)
    if output_path.lower().endswith(".csv"):
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("服务启动中: 加载模型权重...")
    # 仅在服务启动时加载模型,避免重复开销
    _set_deterministic()
    ckpt_path = os.getenv("LIQE_WEIGHTS", "weights/LIQE.pt")
    if not os.path.exists(ckpt_path):
        # 启动阶段抛错可快速暴露缺失权重问题
        raise RuntimeError(f"模型权重不存在: {ckpt_path}")
    print(f"已加载模型权重: {ckpt_path}")
    device = _get_device()
    model = create_model(ckpt_path, device)
    # 通过 app.state 缓存模型与设备,供请求复用
    app.state.device = device
    app.state.model = model
    app.state.ckpt_path = ckpt_path
    yield


app = FastAPI(title="LIQE API", version="1.0", lifespan=lifespan)  # FastAPI 应用实例

@app.get("/health")
def health_check() -> dict:
    # 健康检查接口,可用于负载均衡器或监控系统
    return {"status": "ok", "model_loaded": bool(app.state.model)}


@app.post("/predict", response_model=PredictResult)
async def predict(file: UploadFile = File(...)) -> PredictResult:
    # 单图推理
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供图片文件")
    img = await _read_upload_image(file)
    result = _predict_image(app.state.model, app.state.device, img)
    result.file_name = file.filename
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(files: List[UploadFile] = File(...)) -> BatchPredictResponse:
    # 多图推理
    if not files:
        raise HTTPException(status_code=400, detail="未提供图片文件列表")
    results: List[PredictResult] = []
    for file in files:
        # 每张图片单独解析与推理,避免相互影响
        img = await _read_upload_image(file)
        result = _predict_image(app.state.model, app.state.device, img)
        result.file_name = file.filename
        results.append(result)
    return BatchPredictResponse(results=results)


@app.post("/process/folder", response_model=ProcessFolderResult)
def process_folder(payload: ProcessFolderRequest) -> ProcessFolderResult:
    # 复用原有的批量处理逻辑
    input_path = payload.input_path
    output_path = payload.output_path
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"输入文件夹不存在: {input_path}")

    image_files = [
        f for f in os.listdir(input_path) if f.lower().endswith(VALID_EXTENSIONS)
    ]
    if not image_files:
        raise HTTPException(status_code=400, detail="文件夹中未找到图片")

    results: List[dict] = []
    skipped = 0
    for filename in image_files:
        try:
            img_path = os.path.join(input_path, filename)
            img = Image.open(img_path).convert("RGB")
            pred = _predict_image(app.state.model, app.state.device, img)
            # 结果字典方便写出 CSV 或 Excel
            results.append(
                {
                    "image_name": filename,
                    "quality_score": pred.quality_score,
                    "scene": pred.scene,
                    "distortion": pred.distortion,
                }
            )
        except Exception:  # noqa: BLE001
            # 读取或推理失败的图片计入跳过
            skipped += 1

    if not results:
        raise HTTPException(status_code=400, detail="没有可用的图片结果")

    _save_results(results, output_path)
    return ProcessFolderResult(
        input_path=input_path,
        output_path=output_path,
        total_images=len(image_files),
        processed_images=len(results),
        skipped_images=skipped,
    )


if __name__ == "__main__":
    import uvicorn

    # 直接运行文件时启动服务
    uvicorn.run("api_server:app", host="localhost", port=8000, reload=False)
