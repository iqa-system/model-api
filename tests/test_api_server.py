from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = types.ModuleType("torch")

    def _device(value: str) -> str:
        return value

    @contextlib.contextmanager
    def _no_grad():
        yield

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Cudnn:
        deterministic = False
        benchmark = False

    class _Backends:
        cudnn = _Cudnn()

    torch.device = _device
    torch.manual_seed = lambda seed: None
    torch.cuda = _Cuda()
    torch.backends = _Backends()
    torch.no_grad = _no_grad
    sys.modules["torch"] = torch

from api_server import create_app
from model_adapters import ADAPTER_TYPES, BaseIqaModelAdapter, ModelPrediction


class FakeAdapter(BaseIqaModelAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device: torch.device | None = None

    @property
    def capabilities(self) -> list[str]:
        capabilities = ["quality_score"]
        if not self.adapter_options.get("generic_only"):
            capabilities.extend(["scene", "distortion", "distortions"])
        capabilities.extend(self.adapter_options.get("extra_capabilities", []))
        return capabilities

    def load(self, device: torch.device) -> None:
        if self.adapter_options.get("raise_on_load"):
            raise RuntimeError("boom")
        self.device = device

    def predict(self, image: Image.Image) -> ModelPrediction:
        assert self.device is not None
        quality_score = float(self.adapter_options.get("quality_score", 50.0))
        if self.adapter_options.get("generic_only"):
            return ModelPrediction(
                quality_score=quality_score,
                extra_outputs={"artifact_map": [0.25, 0.75]},
            )

        scene = int(self.adapter_options.get("scene", 1))
        distortion = int(self.adapter_options.get("distortion", 2))
        probability = float(self.adapter_options.get("distortion_prob", 0.9))
        return ModelPrediction(
            quality_score=quality_score,
            scene=scene,
            distortion=distortion,
            distortions=[{"index": distortion, "prob": probability}],
            extra_outputs={"model_echo": self.model_id},
        )


@pytest.fixture(autouse=True)
def register_fake_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ADAPTER_TYPES, "fake", FakeAdapter)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    registry_path = _write_registry(
        tmp_path,
        default_model_id="fake-default",
        models=[
            {
                "id": "fake-default",
                "display_name": "Fake Default",
                "description": "默认测试模型",
                "adapter": "fake",
                "enabled": True,
                "adapter_options": {"quality_score": 61.0, "scene": 3, "distortion": 4},
            },
            {
                "id": "fake-alt",
                "display_name": "Fake Alt",
                "description": "可选测试模型",
                "adapter": "fake",
                "enabled": True,
                "adapter_options": {"quality_score": 44.5, "scene": 8, "distortion": 1},
            },
            {
                "id": "generic-model",
                "display_name": "Generic Model",
                "description": "只返回基础评分和扩展输出",
                "adapter": "fake",
                "enabled": True,
                "adapter_options": {
                    "quality_score": 88.8,
                    "generic_only": True,
                    "extra_capabilities": ["artifact_map"],
                },
            },
        ],
    )

    with TestClient(create_app(registry_path)) as test_client:
        yield test_client


def test_list_models_returns_enabled_models_and_default(client: TestClient) -> None:
    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_model_id"] == "fake-default"
    assert [item["model_id"] for item in payload["models"]] == [
        "fake-default",
        "fake-alt",
        "generic-model",
    ]
    assert payload["models"][0]["is_default"] is True
    assert payload["models"][1]["is_default"] is False
    assert payload["models"][2]["capabilities"] == ["quality_score", "artifact_map"]


def test_predict_uses_default_model_when_model_id_missing(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("default.png", _make_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "fake-default"
    assert payload["quality_score"] == 61.0
    assert payload["scene"] == 3
    assert payload["distortion"] == 4


def test_predict_matches_explicit_default_model(client: TestClient) -> None:
    implicit_response = client.post(
        "/predict",
        files={"file": ("same.png", _make_image_bytes(), "image/png")},
    )
    explicit_response = client.post(
        "/predict",
        data={"model_id": "fake-default"},
        files={"file": ("same.png", _make_image_bytes(), "image/png")},
    )

    assert implicit_response.status_code == 200
    assert explicit_response.status_code == 200
    assert implicit_response.json()["model_id"] == explicit_response.json()["model_id"]
    assert implicit_response.json()["quality_score"] == explicit_response.json()["quality_score"]
    assert implicit_response.json()["scene"] == explicit_response.json()["scene"]
    assert implicit_response.json()["distortion"] == explicit_response.json()["distortion"]


def test_predict_supports_explicit_model_selection(client: TestClient) -> None:
    response = client.post(
        "/predict",
        data={"model_id": "fake-alt"},
        files={"file": ("alt.png", _make_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "fake-alt"
    assert payload["quality_score"] == 44.5
    assert payload["scene"] == 8
    assert payload["extra_outputs"] == {"model_echo": "fake-alt"}


def test_predict_rejects_unknown_model_id(client: TestClient) -> None:
    response = client.post(
        "/predict",
        data={"model_id": "missing-model"},
        files={"file": ("missing.png", _make_image_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert "未知模型" in response.json()["detail"]


def test_predict_supports_generic_adapter_outputs(client: TestClient) -> None:
    response = client.post(
        "/predict",
        data={"model_id": "generic-model"},
        files={"file": ("generic.png", _make_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "generic-model"
    assert payload["quality_score"] == 88.8
    assert payload["scene"] is None
    assert payload["distortion"] is None
    assert payload["distortions"] is None
    assert payload["extra_outputs"] == {"artifact_map": [0.25, 0.75]}


def test_predict_batch_returns_results_for_each_file(client: TestClient) -> None:
    response = client.post(
        "/predict/batch",
        data={"model_id": "fake-alt"},
        files=[
            ("files", ("a.png", _make_image_bytes((255, 0, 0)), "image/png")),
            ("files", ("b.png", _make_image_bytes((0, 255, 0)), "image/png")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 2
    assert [item["file_name"] for item in payload["results"]] == ["a.png", "b.png"]
    assert all(item["model_id"] == "fake-alt" for item in payload["results"])
    assert all(item["quality_score"] == 44.5 for item in payload["results"])


def test_process_folder_endpoint_is_removed(client: TestClient) -> None:
    response = client.post(
        "/process/folder",
        json={"input_path": "D:/data/images", "output_path": "D:/data/output/result.csv"},
    )

    assert response.status_code == 404
    assert "/process/folder" not in client.get("/openapi.json").json()["paths"]


def test_startup_fails_when_weights_missing(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        default_model_id="missing-model",
        models=[
            {
                "id": "missing-model",
                "display_name": "Missing",
                "description": "缺少权重的模型",
                "adapter": "fake",
                "enabled": True,
                "create_weights": False,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="模型权重不存在"):
        with TestClient(create_app(registry_path)):
            pass


def test_startup_fails_when_adapter_load_raises(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        default_model_id="broken-model",
        models=[
            {
                "id": "broken-model",
                "display_name": "Broken",
                "description": "加载时失败的模型",
                "adapter": "fake",
                "enabled": True,
                "adapter_options": {"raise_on_load": True},
            }
        ],
    )

    with pytest.raises(RuntimeError, match="加载模型失败: broken-model"):
        with TestClient(create_app(registry_path)):
            pass


def _make_image_bytes(color: tuple[int, int, int] = (12, 34, 56)) -> bytes:
    image = Image.new("RGB", (32, 32), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_registry(tmp_path: Path, default_model_id: str, models: list[dict]) -> Path:
    registry_models = []
    for model in models:
        model_id = model["id"]
        weights_path = model.get("weights_path", f"weights/{model_id}.pt")
        if model.get("create_weights", True):
            resolved_weights = tmp_path / weights_path
            resolved_weights.parent.mkdir(parents=True, exist_ok=True)
            resolved_weights.write_bytes(b"fake-weights")

        registry_models.append(
            {
                "id": model_id,
                "display_name": model["display_name"],
                "description": model["description"],
                "adapter": model["adapter"],
                "weights_path": weights_path,
                "enabled": model.get("enabled", True),
                "adapter_options": model.get("adapter_options", {}),
            }
        )

    registry_path = tmp_path / "model_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "default_model_id": default_model_id,
                "models": registry_models,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return registry_path
