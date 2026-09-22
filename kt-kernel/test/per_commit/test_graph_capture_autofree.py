"""CPU-only regression tests for graph-capture lifetime decisions."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default")

EXPERTS_BASE_PATH = Path(__file__).resolve().parents[2] / "python" / "experts_base.py"


@pytest.fixture
def experts_base(monkeypatch: pytest.MonkeyPatch):
    """Load experts_base without requiring a compiled kt-kernel extension."""
    package = types.ModuleType("kt_kernel")
    package.kt_kernel_ext = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "kt_kernel", package)

    module_name = "_kt_experts_base_capture_policy_under_test"
    spec = importlib.util.spec_from_file_location(module_name, EXPERTS_BASE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _device(device_type: str):
    if device_type == "cuda":
        return torch.device("cuda")
    # CPU-only PyTorch builds do not register the MUSA private-use device name.
    return types.SimpleNamespace(type=device_type)


def _install_backend_probe(
    monkeypatch: pytest.MonkeyPatch,
    experts_base,
    device_type: str,
    probe,
):
    backend = types.SimpleNamespace()
    if probe is not None:
        backend.is_current_stream_capturing = probe
    monkeypatch.setattr(experts_base.torch, device_type, backend, raising=False)

    def get_device_module(device):
        type_name = (
            device.type if hasattr(device, "type") else str(device).split(":", 1)[0]
        )
        return getattr(experts_base.torch, type_name)

    monkeypatch.setattr(
        experts_base.torch,
        "get_device_module",
        get_device_module,
        raising=False,
    )
    return backend


def test_cpu_is_eager_and_async_forward_args_may_autofree(monkeypatch, experts_base):
    monkeypatch.setattr(experts_base, "_sglang_is_capture_mode", lambda: False)
    monkeypatch.setattr(
        experts_base.torch,
        "get_device_module",
        lambda _device: pytest.fail("CPU must not use an accelerator capture probe"),
        raising=False,
    )

    device = torch.device("cpu")
    assert experts_base._graph_capture_active(device) is False
    assert experts_base._forward_task_autofree(False, device) is True
    assert experts_base._forward_task_autofree(True, device) is True
    experts_base._wait_device(device)


@pytest.mark.parametrize("device_type", ["cuda", "musa"])
@pytest.mark.parametrize("capturing", [False, True])
def test_accelerator_probe_controls_async_autofree(
    monkeypatch,
    experts_base,
    device_type,
    capturing,
):
    monkeypatch.setattr(experts_base, "_sglang_is_capture_mode", lambda: False)
    _install_backend_probe(
        monkeypatch,
        experts_base,
        device_type,
        lambda: capturing,
    )

    device = _device(device_type)
    assert experts_base._graph_capture_active(device) is capturing
    assert experts_base._forward_task_autofree(False, device) is (not capturing)
    # Synchronous submit owns no graph callback, even while another capture is active.
    assert experts_base._forward_task_autofree(True, device) is True


def test_sglang_capture_takes_priority_over_backend_probe(monkeypatch, experts_base):
    probe_calls = 0

    def backend_probe():
        nonlocal probe_calls
        probe_calls += 1
        return False

    monkeypatch.setattr(experts_base, "_sglang_is_capture_mode", lambda: True)
    _install_backend_probe(monkeypatch, experts_base, "musa", backend_probe)

    device = _device("musa")
    assert experts_base._graph_capture_active(device) is True
    assert experts_base._forward_task_autofree(False, device) is False
    assert probe_calls == 0


@pytest.mark.parametrize("device_type", ["cuda", "musa"])
def test_wait_device_synchronizes_eager_accelerators(
    monkeypatch,
    experts_base,
    device_type,
):
    synchronized = []
    monkeypatch.setattr(experts_base, "_sglang_is_capture_mode", lambda: False)
    backend = _install_backend_probe(
        monkeypatch,
        experts_base,
        device_type,
        lambda: False,
    )
    backend.synchronize = synchronized.append

    device = _device(device_type)
    experts_base._wait_device(device)

    assert synchronized == [device]


@pytest.mark.parametrize("device_type", ["cuda", "musa"])
def test_wait_device_does_not_synchronize_during_capture(
    monkeypatch,
    experts_base,
    device_type,
):
    monkeypatch.setattr(experts_base, "_sglang_is_capture_mode", lambda: False)
    backend = _install_backend_probe(
        monkeypatch,
        experts_base,
        device_type,
        lambda: True,
    )
    backend.synchronize = lambda _device: pytest.fail(
        "captured streams must not be synchronized"
    )

    experts_base._wait_device(_device(device_type))


@pytest.mark.parametrize("device_type", ["cuda", "musa"])
@pytest.mark.parametrize("failure", ["missing", "raises"])
def test_unavailable_accelerator_probe_fails_closed(
    monkeypatch,
    experts_base,
    device_type,
    failure,
):
    monkeypatch.setattr(experts_base, "_sglang_is_capture_mode", lambda: False)

    probe = None
    if failure == "raises":

        def probe():
            raise RuntimeError("capture state is unavailable")

    _install_backend_probe(monkeypatch, experts_base, device_type, probe)

    device = _device(device_type)
    assert experts_base._graph_capture_active(device) is True
    assert experts_base._forward_task_autofree(False, device) is False
    assert experts_base._forward_task_autofree(True, device) is True


@pytest.mark.parametrize("device_type", ["cuda", "musa"])
@pytest.mark.parametrize("failure", ["missing", "raises"])
def test_wait_device_rejects_unknown_capture_state(
    monkeypatch,
    experts_base,
    device_type,
    failure,
):
    monkeypatch.setattr(experts_base, "_sglang_is_capture_mode", lambda: False)

    probe = None
    if failure == "raises":

        def probe():
            raise RuntimeError("capture state is unavailable")

    backend = _install_backend_probe(
        monkeypatch,
        experts_base,
        device_type,
        probe,
    )
    backend.synchronize = lambda _device: pytest.fail(
        "unknown capture state must not be synchronized"
    )

    with pytest.raises(RuntimeError, match="capture state is unavailable"):
        experts_base._wait_device(_device(device_type))
