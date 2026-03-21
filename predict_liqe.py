import torch
import numpy as np
import random
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.LIQE import LIQE
from torchvision.transforms import ToTensor

def batch_process_liqe(img_dir, ckpt_path, output_file):
    # 1. 设置随机种子和设备
    seed = 20200626
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载模型
    print(f"正在加载 LIQE 模型 (ckpt: {ckpt_path})...")
    try:
        model = LIQE(ckpt_path, device)
        model.eval() # 确保是评估模式
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 扫描文件夹
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    if not os.path.exists(img_dir):
        print(f"错误: 输入文件夹不存在 {img_dir}")
        return

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"文件夹 {img_dir} 中未找到图片")
        return

    results = []

    # 4. 批量推理
    print(f"开始处理文件夹: {img_dir}，共 {len(image_files)} 张图片")
    with torch.no_grad():
        for filename in tqdm(image_files, desc="批处理图片..."):
            try:
                img_path = os.path.join(img_dir, filename)
                img = Image.open(img_path).convert('RGB') # 确保是一致的颜色模式

                # 预处理
                img_tensor = ToTensor()(img).unsqueeze(0)

                # 推理
                q, s, d = model(img_tensor)

                results.append({
                    'image_name': filename,
                    'quality_score': q.item(),
                    'scene': int(s.item()),
                    'distortion': int(d.item())
                })
            except Exception as e:
                print(f"\n跳过损坏图片 {filename}: {e}")

    # 5. 保存结果
    if not results:
        print("没有结果生成")
        return

    df = pd.DataFrame(results)

    # 确保存放结果的文件夹存在
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
    else:
        df.to_excel(output_file, index=False)

    print(f"\n处理完成！结果已保存至: {output_file}")

if __name__ == '__main__':
    print("STARTING LIQE BATCH PROCESS")
    # === 请修改以下配置 ===
    INPUT_FOLDER = 'test_imgs'         # 图片存放文件夹
    LIQE_WEIGHTS = 'weights/LIQE.pt' # LIQE 模型权重路径
    OUTPUT_PATH = 'out/liqe_results.csv'   # 输出文件名 (支持 .csv 或 .xlsx)

    batch_process_liqe(INPUT_FOLDER, LIQE_WEIGHTS, OUTPUT_PATH)
