from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

import pytest

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

from model_adapters import PyiqaMetricAdapter
from model_registry import _resolve_weights_path


class FakeMetric:
    # 这个假对象用来模拟 pyiqa.create_metric 的返回值.
    # 我们只保留当前测试真正需要的几个行为, 这样测试更聚焦.
    def __init__(self) -> None:
        self.eval_called = False
        self.load_weights_calls: list[tuple[str, str | None]] = []

    def eval(self) -> None:
        self.eval_called = True

    def load_weights(self, weights_path: str, weight_keys: str | None = "params") -> None:
        self.load_weights_calls.append((weights_path, weight_keys))


def test_resolve_weights_path_prefers_project_weights_directory_for_bare_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # 当配置里只写文件名时, 服务应该优先到项目 weights 目录里找.
    # 这样配置会更简洁, 也符合这次 "统一放到 weights 目录" 的约定.
    fake_service_root = tmp_path / "service"
    fake_weights_root = fake_service_root / "weights"
    fake_weights_root.mkdir(parents=True)
    weight_file = fake_weights_root / "demo-model.pth"
    weight_file.write_bytes(b"demo")

    monkeypatch.setattr("model_registry.SERVICE_ROOT", fake_service_root)
    monkeypatch.setattr("model_registry.WEIGHTS_ROOT", fake_weights_root)

    config_path = tmp_path / "config" / "model_registry.json"
    resolved = _resolve_weights_path(config_path, "demo-model.pth")

    assert resolved == weight_file.resolve()


def test_pyiqa_adapter_constructor_path_injects_local_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # 这个测试验证默认策略下会把项目内权重路径显式传给 pyiqa.create_metric.
    # 只要这一步成立, pyiqa 就不会再退回到用户目录缓存或联网下载.
    captured: dict[str, object] = {}
    fake_metric = FakeMetric()

    def fake_create_metric(metric_name: str, device=None, **kwargs):
        captured["metric_name"] = metric_name
        captured["device"] = device
        captured["kwargs"] = kwargs
        return fake_metric

    fake_pyiqa = types.SimpleNamespace(create_metric=fake_create_metric)
    monkeypatch.setitem(sys.modules, "pyiqa", fake_pyiqa)

    weights_path = tmp_path / "weights" / "musiq.pth"
    adapter = PyiqaMetricAdapter(
        model_id="musiq_koniq",
        display_name="MUSIQ",
        description="test",
        weights_path=weights_path,
        adapter_options={"metric_name": "musiq"},
    )

    device = torch.device("cpu")
    adapter.load(device)

    assert captured["metric_name"] == "musiq"
    assert captured["device"] == device
    assert captured["kwargs"] == {"pretrained_model_path": str(weights_path)}
    assert fake_metric.eval_called is True


def test_pyiqa_adapter_manual_load_strategy_calls_load_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # 对于未来不支持 constructor_path 的 pyiqa 模型,
    # 我们也保留了 "先创建, 再手动 load_weights" 的兜底方案.
    captured_kwargs: dict[str, object] = {}
    fake_metric = FakeMetric()

    def fake_create_metric(metric_name: str, device=None, **kwargs):
        captured_kwargs["metric_name"] = metric_name
        captured_kwargs["device"] = device
        captured_kwargs["kwargs"] = kwargs
        return fake_metric

    fake_pyiqa = types.SimpleNamespace(create_metric=fake_create_metric)
    monkeypatch.setitem(sys.modules, "pyiqa", fake_pyiqa)

    weights_path = tmp_path / "weights" / "topiq.pth"
    adapter = PyiqaMetricAdapter(
        model_id="topiq_nr_koniq",
        display_name="TOPIQ-NR",
        description="test",
        weights_path=weights_path,
        adapter_options={
            "metric_name": "topiq_nr",
            "load_strategy": "manual_load_weights",
            "metric_options": {"test_img_size": 384},
            "weight_keys": "params",
        },
    )

    adapter.load(torch.device("cpu"))

    assert captured_kwargs["metric_name"] == "topiq_nr"
    assert captured_kwargs["kwargs"] == {"test_img_size": 384}
    assert fake_metric.load_weights_calls == [(str(weights_path), "params")]
    assert fake_metric.eval_called is True
