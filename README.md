# model-api
模型推理服务_python
- 参考文献: https://github.com/zwx8981/LIQE

## FastAPI 服务

### 依赖安装

```powershell
pip install -r requirements.txt
```

### 启动服务

```powershell
python api_server.py
```

服务启动后：
- `POST /predict`
- `POST /predict/batch`