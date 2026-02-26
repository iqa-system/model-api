import os
import torch
from PIL import Image

from api_server import create_model, _get_device, _predict_image


def main() -> None:
    # 本地冒烟测试,验证模型能正常推理
    ckpt_path = os.getenv("LIQE_WEIGHTS", "weights/LIQE.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"模型权重不存在: {ckpt_path}")

    device = _get_device()
    model = create_model(ckpt_path, device)

    img_path = os.path.join("test_imgs", "toothless.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"测试图片不存在: {img_path}")

    img = Image.open(img_path).convert("RGB")
    result = _predict_image(model, device, img)
    print(result.model_dump())


if __name__ == "__main__":
    main()
