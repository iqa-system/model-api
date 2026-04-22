from __future__ import annotations

# 这个文件是离线批处理脚本.
# 它不属于 HTTP 服务接口, 但会复用同一套模型注册表和 adapter 逻辑.
# 这样服务端和脚本端不会各自维护一套 "模型选择" 规则.

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from model_registry import ModelRegistry


# 默认配置和服务端保持一致.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_REGISTRY_PATH = BASE_DIR / "config" / "model_registry.json"
# 这里定义允许扫描的图片扩展名.
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# 允许读取被截断图片, 和服务端保持一致.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _set_deterministic(seed: int = 20200626) -> None:
    # 固定随机性, 让离线脚本结果更稳定, 便于和服务端对比.
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_device() -> torch.device:
    # 有 GPU 就用 GPU, 否则用 CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_results(results: list[dict], output_path: Path) -> None:
    # 如果输出目录不存在, 先自动创建.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    # 用文件后缀决定导出格式.
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False)


def _load_registry(registry_path: Path) -> ModelRegistry:
    # 这个函数集中负责 "固定随机性 + 选择设备 + 加载注册表".
    # 单图预测和批量预测都复用它.
    _set_deterministic()
    device = _get_device()
    print(f"正在加载模型注册表: {registry_path}")
    return ModelRegistry.from_path(registry_path, device)


def predict_single_image(
    img_path: str | Path,
    model_id: str | None = None,
    registry_path: str | Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> dict | None:
    # 这个函数适合给其它 Python 代码调用.
    # 输入一张图片路径, 返回一个字典结果, 不直接写文件.
    registry = _load_registry(Path(registry_path))
    image_path = Path(img_path)
    if not image_path.exists():
        print(f"图片不存在: {image_path}")
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        # registry.predict 会完成默认模型回退和模型查找.
        resolved_model_id, prediction = registry.predict(model_id, img)
        return {
            "image_name": image_path.name,
            "model_id": resolved_model_id,
            "quality_score": prediction.quality_score,
            "scene": prediction.scene,
            "distortion": prediction.distortion,
            "distortions": prediction.distortions,
            "extra_outputs": prediction.extra_outputs,
        }
    except Exception as exc:  # noqa: BLE001
        # 离线脚本这里选择打印错误并返回 None, 而不是抛出 HTTP 异常.
        # 因为它不是 Web 接口, 更适合命令行风格的错误反馈.
        print(f"处理图片 {image_path} 失败: {exc}")
        return None


def batch_process_images(
    img_dir: str | Path,
    output_file: str | Path,
    model_id: str | None = None,
    registry_path: str | Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> None:
    # 批量处理函数的主要流程是:
    # 1. 校验输入目录.
    # 2. 扫描图片文件.
    # 3. 加载模型注册表.
    # 4. 逐张图片执行推理.
    # 5. 把结果保存成 CSV 或 Excel.
    input_dir = Path(img_dir)
    output_path = Path(output_file)
    if not input_dir.exists():
        print(f"错误: 输入文件夹不存在 {input_dir}")
        return

    # 只挑出符合扩展名规则的图片文件.
    image_files = [path for path in input_dir.iterdir() if path.suffix.lower() in VALID_EXTENSIONS]
    if not image_files:
        print(f"文件夹 {input_dir} 中未找到图片")
        return

    registry = _load_registry(Path(registry_path))
    try:
        # 先把 model_id 解析一次, 避免循环中重复校验.
        resolved_model_id = registry.resolve_model_id(model_id)
    except ValueError as exc:
        print(exc)
        return

    print(
        f"开始处理文件夹: {input_dir}，共 {len(image_files)} 张图片，"
        f"使用模型: {resolved_model_id}"
    )
    results: list[dict] = []
    for image_path in tqdm(image_files, desc="批处理图片..."):
        try:
            img = Image.open(image_path).convert("RGB")
            # 这里传入 resolved_model_id, 表示整个批次固定使用同一个模型.
            _, prediction = registry.predict(resolved_model_id, img)
            results.append(
                {
                    "image_name": image_path.name,
                    "model_id": resolved_model_id,
                    "quality_score": prediction.quality_score,
                    "scene": prediction.scene,
                    "distortion": prediction.distortion,
                    # 复杂字段先转成 JSON 字符串, 这样写入 CSV 和 Excel 都更稳定.
                    "distortions": json.dumps(
                        prediction.distortions or [],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "extra_outputs": json.dumps(
                        prediction.extra_outputs,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            # 单张图失败时只跳过当前文件, 不中断整个批处理流程.
            print(f"\n跳过图片 {image_path.name}: {exc}")

    if not results:
        print("没有结果生成")
        return

    _save_results(results, output_path)
    print(f"\n处理完成！结果已保存至: {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    # argparse 用来定义命令行参数.
    # 这样用户可以通过命令行传入输入目录, 输出路径, 模型 ID 等信息.
    parser = argparse.ArgumentParser(description="离线批量执行图像质量评估")
    parser.add_argument("--input-folder", default="test_imgs", help="图片文件夹路径")
    parser.add_argument(
        "--output-path",
        default="out/liqe_results.csv",
        help="输出文件路径，支持 .csv 或 .xlsx",
    )
    parser.add_argument("--model-id", default=None, help="可选模型 ID，未传则使用默认模型")
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_MODEL_REGISTRY_PATH),
        help="模型注册表配置文件路径",
    )
    return parser


if __name__ == "__main__":
    # 直接运行脚本时, 从命令行解析参数后开始执行批量处理.
    args = _build_parser().parse_args()
    batch_process_images(
        img_dir=args.input_folder,
        output_file=args.output_path,
        model_id=args.model_id,
        registry_path=args.registry_path,
    )
