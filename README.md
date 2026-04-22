# model-api

图像质量评估模型服务，基于 FastAPI 提供推理接口。

- 支持单图推理：`POST /predict`
- 支持多图批量推理：`POST /predict/batch`
- 支持模型发现：`GET /models`
- 默认模型和可用模型列表由 `config/model_registry.json` 管理

## 依赖安装

```powershell
pip install -r requirements.txt
```

## 启动服务

```powershell
python api_server.py
```

服务启动时会预加载 `config/model_registry.json` 中所有启用的模型。

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
      "description": "当前系统默认使用的 LIQE 权重。",
      "adapter": "liqe",
      "weights_path": "weights/LIQE.pt",
      "enabled": true
    }
  ]
}
```

新增模型的方式：

1. 将权重文件放入服务器。
2. 在 `config/model_registry.json` 中新增模型项。
3. 重启服务。

## 接口说明

### `GET /models`

返回当前启用的模型列表、默认模型 ID、模型展示名和能力说明。

### `POST /predict`

`multipart/form-data` 参数：

- `file`: 图片文件
- `model_id`: 可选，未传时使用默认模型

### `POST /predict/batch`

`multipart/form-data` 参数：

- `files`: 多个图片文件
- `model_id`: 可选，未传时使用默认模型

## 离线批处理脚本

`predict_liqe.py` 保留为本地离线工具，也会复用同一份模型注册表配置：

```powershell
python predict_liqe.py --input-folder test_imgs --output-path out/liqe_results.csv --model-id liqe_mix
```
