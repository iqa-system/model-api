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

## 权重目录

项目约定统一使用根目录下的 `weights/` 保存模型权重。

- `LIQE.pt`
- `musiq_koniq_ckpt-e95806b9.pth`
- `cfanet_nr_koniq_res50-9a73138b.pth`

如果 `weights_path` 只写文件名，服务会优先到 `weights/` 目录中查找。

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
        "metric_name": "topiq_nr"
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
