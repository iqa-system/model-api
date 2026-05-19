from itertools import product
import os
from pathlib import Path

import warnings
# 忽略 pkg_resources 的弃用警告
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from torchvision.transforms import Normalize

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 失真类型概率阈值, 超过该阈值的失真类型将被保留; 如果没有任何失真类型超过该阈值, 则保留概率最高的一个失真类型
DISTORTION_PROB_THRESHOLD = 0.10

# LIQE 底层使用 OpenAI CLIP 的 ViT-B/32 作为基础视觉语言模型.
# openai-clip 默认会把这个文件下载到 ~/.cache/clip/ViT-B-32.pt.
# 为了让服务器离线部署更稳定, 这里显式要求从项目 weights 目录读取该文件.
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_MODEL_FILE_NAME = "ViT-B-32.pt"

scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
scene2label = {'animal':0, 'cityscape':1, 'human':2, 'indoor':3, 'landscape':4, 'night':5, 'plant':6, 'still_life':7, 'others':8}

map2label = {'jpeg2000 compression':0, 'jpeg compression':1, 'noise':2, 'blur':3, 'color':4,
             'contrast':5, 'overexposure':6, 'underexposure':7, 'spatial':8, 'quantization':9, 'other':10}
dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
            'underexposure', 'spatial', 'quantization', 'other']

qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

type2label = {'jpeg2000 compression':0, 'jpeg compression':1, 'white noise':2, 'gaussian blur':3, 'fastfading':4, 'fnoise':5, 'contrast':6, 'lens':7, 'motion':8,
              'diffusion':9, 'shifting':10, 'color quantization':11, 'oversaturation':12, 'desaturation':13,
              'white with color':14, 'impulse':15, 'multiplicative':16, 'white noise with denoise':17, 'brighten':18,
              'darken':19, 'shifting the mean':20, 'jitter':21, 'noneccentricity patch':22, 'pixelate':23,
              'quantization':24, 'color blocking':25, 'sharpness':26, 'realistic blur':27, 'realistic noise':28,
              'underexposure':29, 'overexposure':30, 'realistic contrast change':31, 'other realistic':32}

dist_map = {'jpeg2000 compression':'jpeg2000 compression', 'jpeg compression':'jpeg compression',
                   'white noise':'noise', 'gaussian blur':'blur', 'fastfading': 'jpeg2000 compression', 'fnoise':'noise',
                   'contrast':'contrast', 'lens':'blur', 'motion':'blur', 'diffusion':'color', 'shifting':'blur',
                   'color quantization':'quantization', 'oversaturation':'color', 'desaturation':'color',
                   'white with color':'noise', 'impulse':'noise', 'multiplicative':'noise',
                   'white noise with denoise':'noise', 'brighten':'overexposure', 'darken':'underexposure', 'shifting the mean':'other',
                   'jitter':'spatial', 'noneccentricity patch':'spatial', 'pixelate':'spatial', 'quantization':'quantization',
                   'color blocking':'spatial', 'sharpness':'contrast', 'realistic blur':'blur', 'realistic noise':'noise',
                   'underexposure':'underexposure', 'overexposure':'overexposure', 'realistic contrast change':'contrast', 'other realistic':'other'}

dists = ['jpeg2000 compression', 'jpeg compression', 'white noise', 'gaussian blur', 'fastfading', 'fnoise', 'contrast', 'lens', 'motion', 'diffusion', 'shifting',
         'color quantization', 'oversaturation', 'desaturation', 'white with color', 'impulse', 'multiplicative',
         'white noise with denoise', 'brighten', 'darken', 'shifting the mean', 'jitter', 'noneccentricity patch',
         'pixelate', 'quantization', 'color blocking', 'sharpness', 'realistic blur', 'realistic noise',
         'underexposure', 'overexposure', 'realistic contrast change', 'other realistic']


def _resolve_clip_download_root(ckpt_path: Path) -> Path:
    # 默认把 CLIP 基础权重和 LIQE.pt 放在同一个目录.
    # 对当前项目来说, 这个目录通常就是项目根目录下的 weights/.
    # 如果部署时想把 CLIP 权重单独放到其它目录, 可以通过 LIQE_CLIP_WEIGHTS_DIR 覆盖.
    configured_root = os.getenv("LIQE_CLIP_WEIGHTS_DIR")
    clip_root = Path(configured_root).expanduser().resolve() if configured_root else ckpt_path.parent
    clip_weight_path = clip_root / CLIP_MODEL_FILE_NAME

    # 这里主动检查文件是否存在, 是为了避免 openai-clip 在服务器上悄悄联网下载.
    # 如果文件缺失, 直接报出应该放置的路径, 排查起来比等待下载失败更清楚.
    if not clip_weight_path.exists():
        raise FileNotFoundError(
            "CLIP 基础权重不存在: "
            f"{clip_weight_path}. "
            "请把已下载的 ViT-B-32.pt 复制到该目录, "
            "或通过环境变量 LIQE_CLIP_WEIGHTS_DIR 指定 CLIP 权重目录."
        )

    return clip_root

class LIQE(nn.Module):
    def __init__(self, ckpt, device):
        super(LIQE, self).__init__()
        ckpt_path = Path(ckpt).resolve()
        clip_download_root = _resolve_clip_download_root(ckpt_path)
        self.model, preprocess = clip.load(
            CLIP_MODEL_NAME,
            device=device,
            jit=False,
            download_root=str(clip_download_root),
        )
        checkpoint = torch.load(ckpt_path, map_location=device)


        self.model.load_state_dict(checkpoint)

        joint_texts = torch.cat(
            [clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
             in product(qualitys, scenes, dists_map)]).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(joint_texts)
            self.text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        self.step = 32
        self.num_patch = 15
        self.normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.size(0)
        x = self.normalize(x)
        x = x.unfold(2, 224, self.step).unfold(3, 224, self.step).permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, 3, 224, 224)

        sel_step = x.size(1) // self.num_patch
        sel = torch.zeros(self.num_patch)
        for i in range(self.num_patch):
            sel[i] = sel_step * i
        sel = sel.long()
        x = x[:, sel, ...]
        x = x.reshape(batch_size*self.num_patch, x.shape[2], x.shape[3], x.shape[4])
        #num_patch = x.size(1)
        #x = x.view(-1, x.size(2), x.size(3), x.size(4))

        image_features = self.model.encode_image(x)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ self.text_features.t()

        logits_per_image = logits_per_image.view(batch_size, self.num_patch, -1)
        logits_per_image = logits_per_image.mean(1)
        logits_per_image = F.softmax(logits_per_image, dim=1)

        logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
        logits_quality = logits_per_image.sum(3).sum(2)

        similarity_scene = logits_per_image.sum(3).sum(1)
        similarity_distortion = logits_per_image.sum(1).sum(1)
        scene_index = similarity_scene.argmax(dim=1)

        # print('similarity_scene = ', similarity_scene)

        # 保留所有概率超过阈值的失真类型; 如果没有任何失真类型超过该阈值, 则保留概率最高的一个失真类型
        distortions = []
        distortion_top1 = []
        for sample_probs in similarity_distortion:
            candidates = []
            for idx, prob in enumerate(sample_probs.tolist()):
                prob = float(prob)
                if prob >= DISTORTION_PROB_THRESHOLD:
                    candidates.append((idx, prob))

            if not candidates:
                top_idx = int(sample_probs.argmax().item())
                candidates = [(top_idx, float(sample_probs[top_idx].item()))]
            else:
                candidates.sort(key=lambda x: x[1], reverse=True)

            selected = [{"index": idx, "prob": round(prob, 4)} for idx, prob in candidates]
            distortions.append(selected)
            distortion_top1.append(selected[0]["index"])

        distortion_top1 = torch.tensor(distortion_top1, dtype=torch.long, device=similarity_distortion.device)

        quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                             4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

        return quality, scene_index, distortion_top1, distortions

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt = '../weights/LIQE.pt'
    liqe = LIQE(ckpt, device)

    x = torch.randn(1,3,512,512).to(device)
    q, s, d, ds = liqe(x)
