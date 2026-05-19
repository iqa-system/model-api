# model-api

图像质量评估模型服务，基于 FastAPI 提供推理接口。

- 支持单图推理：`POST /predict`
- 支持多图批量推理：`POST /predict/batch`
- 支持模型发现：`GET /models`
- 默认模型和可用模型列表由 `config/model_registry.json` 管理
- 当前默认接入了 `LIQE`、`MUSIQ`、`TOPIQ-NR`

## 依赖安装

```powershell
pip install -r requirements.txt
```

## 启动服务

```powershell
python api_server.py
```

服务启动时会预加载 `config/model_registry.json` 中所有启用的模型。

## Linux 后台运行

Debian 服务器部署时可以使用 `scripts/` 目录下的脚本一键启停服务。

首次上传到服务器后, 先赋予脚本执行权限:

```bash
chmod +x scripts/*.sh
```

后台启动服务:

```bash
./scripts/start_model_api.sh
```

查看运行状态和最近日志:

```bash
./scripts/status_model_api.sh
```

停止服务:

```bash
./scripts/stop_model_api.sh
```

默认配置:

- 服务监听地址: `0.0.0.0`
- 服务监听端口: `8000`
- Conda 环境名: `model-api`
- 日志文件: `logs/model-api.log`
- PID 文件: `logs/model-api.pid`

如果需要临时修改端口, 可以这样启动:

```bash
IQA_API_PORT=9000 ./scripts/start_model_api.sh
```

如果需要处理高分辨率图片, 可以通过环境变量调整推理前缩放限制:

```bash
IQA_MAX_INFERENCE_LONG_SIDE=2048 ./scripts/start_model_api.sh
```

说明:

- 上传文件体积上限由 Java 后端 SpringBoot 统一控制.
- `IQA_MAX_INFERENCE_LONG_SIDE`: 推理前允许的图片最长边, 默认 `2048` 像素. 超过后会等比例缩小, 以避免高分辨率图片导致内存或显存溢出.

## 权重目录

项目约定统一使用根目录下的 `weights/` 保存模型权重。

- `LIQE.pt`
- `ViT-B-32.pt`
- `musiq_koniq_ckpt-e95806b9.pth`
- `cfanet_nr_koniq_res50-9a73138b.pth`

如果 `weights_path` 只写文件名，服务会优先到 `weights/` 目录中查找。

`ViT-B-32.pt` 是 LIQE 依赖的 OpenAI CLIP 基础权重. `openai-clip` 默认会把它下载到 `~/.cache/clip/ViT-B-32.pt`. 为了方便离线部署, 当前服务默认要求它和 `LIQE.pt` 放在同一个 `weights/` 目录中. 如果你想把 CLIP 权重放到其它目录, 可以通过环境变量 `LIQE_CLIP_WEIGHTS_DIR` 指定.

从旧服务器缓存复制到项目权重目录的示例:

```bash
cp ~/.cache/clip/ViT-B-32.pt /home/model-api/weights/ViT-B-32.pt
```

## 模型配置

默认配置文件：`config/model_registry.json`

示例字段：

```json
{
  "default_model_id": "liqe_default",
  "models": [
    {
      "id": "liqe_default",
      "display_name": "LIQE Default",
      "description": "基于 vision-language multitask 的 NR-IQA 模型. 强项是可解释性和泛化能力, 当前服务还能额外输出 scene 与 distortion. 论文报告其在多个 IQA 数据集上优于当时 SOTA, 并且跨数据集对齐能力更强.",
      "adapter": "liqe",
      "weights_path": "LIQE.pt",
      "enabled": true
    },
    {
      "id": "musiq_koniq",
      "display_name": "MUSIQ KonIQ-10k",
      "description": "多尺度 Transformer 无参考图像质量模型. 能直接处理原生分辨率和不同长宽比图像, 对高分辨率细节和整体构图更友好. 论文在 PaQ-2-PiQ, SPAQ, KonIQ-10k 等大规模数据集上报告了 SOTA 表现.",
      "adapter": "pyiqa",
      "weights_path": "musiq_koniq_ckpt-e95806b9.pth",
      "enabled": true,
      "adapter_options": {
        "metric_name": "musiq"
      }
    },
    {
      "id": "topiq_nr_koniq",
      "display_name": "TOPIQ-NR KonIQ",
      "description": "语义引导的 coarse-to-fine 无参考图像质量模型. 强项是让高层语义信息引导局部失真关注, 在自然场景图像上通常兼顾精度与效率. 论文报告其在多数公开 FR 与 NR 基准上达到 better or competitive 表现, 且效率明显更高.",
      "adapter": "pyiqa",
      "weights_path": "cfanet_nr_koniq_res50-9a73138b.pth",
      "enabled": true,
      "adapter_options": {
        "metric_name": "topiq_nr",
        "metric_options": {
          "backbone_pretrain": false
        }
      }
    }
  ]
}
```

新增模型的方式：

1. 将权重文件放入项目 `weights/` 目录。
2. 在 `config/model_registry.json` 中新增模型项。
3. 重启服务。

## 模型特点

- `LIQE`: 更偏可解释和泛化. 如果你希望除了总分之外, 还能看到场景类别和失真类别, 它最合适.
- `MUSIQ`: 更偏多尺度细节和构图感知. 对高分辨率, 构图变化大, 长宽比不固定的自然图像更友好.
- `TOPIQ-NR`: 更偏语义引导下的通用质量打分. 在自然场景图像上通常兼顾效果和推理效率, 适合做默认候选或快速对比.

LIQE 内部会在 Tensor 阶段自动处理小尺寸图片. 如果图片任意一边小于 `224` 像素, 会使用边缘像素复制补边到满足 `224x224` patch 窗口要求. 对于原本已经大于等于 `224x224` 的图片, LIQE 仍保留原始 patch 选择算法, 以尽量保持评分一致.

部署到离线或内网服务器时, `TOPIQ-NR` 配置里的 `metric_options.backbone_pretrain=false` 可以避免 `timm` 为 ResNet50 骨干网络继续访问 HuggingFace 下载额外权重. 服务仍会加载 `weights/cfanet_nr_koniq_res50-9a73138b.pth` 作为 TOPIQ 主权重.

## 接口说明

### `GET /models`

返回当前启用的模型列表、默认模型 ID、模型展示名、模型特点说明和能力字段。

当前默认模型 ID:

- `liqe_default`

当前示例模型 ID:

- `liqe_default`
- `musiq_koniq`
- `topiq_nr_koniq`

### `POST /predict`

`multipart/form-data` 参数：

- `file`: 图片文件
- `model_id`: 可选，未传时使用默认模型

返回字段说明：

- 所有模型都会返回 `model_id` 和 `quality_score`
- `LIQE` 还会返回 `scene`、`distortion`、`distortions`
- `MUSIQ` 和 `TOPIQ-NR` 当前只返回 `quality_score`

### `POST /predict/batch`

`multipart/form-data` 参数：

- `files`: 多个图片文件
- `model_id`: 可选，未传时使用默认模型

静态 OpenAPI 文档见 [openapi.yaml](/D:/code/repositories/iqa-system/model-api/openapi.yaml:1)。

## 离线批处理脚本

`predict_liqe.py` 保留为本地离线工具，也会复用同一份模型注册表配置：

```powershell
python predict_liqe.py --input-folder test_imgs --output-path out/liqe_results.csv --model-id liqe_mix
```
