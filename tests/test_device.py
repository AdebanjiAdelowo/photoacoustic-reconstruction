"""Device resolution, determinism settings and device-safe conversions. No CUDA is needed: the
availability probes and CUDA calls are replaced where a test depends on them."""
import glob
import os
import re

import numpy as np
import pytest
import torch

from src import device as dev
from src.reconstruction_net import ReconstructionUNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _available(monkeypatch, cuda, mps):
    monkeypatch.setattr(dev, "cuda_available", lambda: cuda)
    monkeypatch.setattr(dev, "mps_available", lambda: mps)


def test_cpu_always_resolves(no_accelerators):
    assert dev.resolve_device("cpu") == torch.device("cpu")


def test_explicit_cuda_fails_clearly_when_unavailable(monkeypatch):
    _available(monkeypatch, cuda=False, mps=True)
    with pytest.raises(dev.DeviceUnavailableError, match="--device cuda"):
        dev.resolve_device("cuda")


def test_explicit_mps_fails_clearly_when_unavailable(monkeypatch):
    _available(monkeypatch, cuda=True, mps=False)
    with pytest.raises(dev.DeviceUnavailableError, match="--device mps"):
        dev.resolve_device("mps")


@pytest.mark.parametrize("cuda, mps, expected", [(True, True, "cuda"), (True, False, "cuda"),
                                                 (False, True, "mps"), (False, False, "cpu")])
def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    _available(monkeypatch, cuda, mps)
    assert dev.resolve_device("auto").type == expected


def test_explicit_choices_resolve_to_exactly_that_device(monkeypatch):
    _available(monkeypatch, cuda=True, mps=True)
    assert dev.resolve_device("cuda").type == "cuda"
    assert dev.resolve_device("mps").type == "mps"
    assert dev.resolve_device("cpu").type == "cpu"


def test_unknown_choice_is_rejected():
    with pytest.raises(ValueError):
        dev.resolve_device("gpu")


def test_synchronize_calls_the_backend_of_the_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("cuda"))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))
    dev.synchronize(torch.device("cpu"))
    assert calls == []
    dev.synchronize(torch.device("cuda"))
    dev.synchronize("mps")
    assert calls == ["cuda", "mps"]


def test_cuda_determinism_settings(monkeypatch, restore_torch_flags):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.backends.cudnn.benchmark = True
    settings = dev.configure_determinism(torch.device("cuda"))
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert settings["deterministic_algorithms"] and settings["tf32_matmul"] is False


def test_cpu_determinism_and_mps_left_alone(restore_torch_flags):
    torch.use_deterministic_algorithms(False)
    assert dev.configure_determinism(torch.device("mps"))["deterministic_algorithms"] is False
    assert not torch.are_deterministic_algorithms_enabled()
    assert dev.configure_determinism(torch.device("cpu"))["deterministic_algorithms"] is True
    assert torch.are_deterministic_algorithms_enabled()


def test_train_get_device_keeps_the_historical_cpu_behaviour(monkeypatch, restore_torch_flags):
    from scripts.train import get_device

    torch.use_deterministic_algorithms(False)
    _available(monkeypatch, cuda=False, mps=False)
    assert get_device("auto").type == "cpu" and not torch.are_deterministic_algorithms_enabled()
    assert get_device("cpu").type == "cpu" and torch.are_deterministic_algorithms_enabled()
    with pytest.raises(dev.DeviceUnavailableError):
        get_device("cuda")


def test_to_numpy_detaches_and_keeps_values():
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    out = dev.to_numpy(t * 2)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert np.array_equal(out, np.arange(6, dtype=np.float32).reshape(2, 3) * 2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an accelerator; MPS is the one available locally")
def test_to_numpy_and_state_dict_from_an_accelerator():
    model = ReconstructionUNet(base_features=16).to("mps").eval()
    x = torch.zeros(1, 1, 64, 64, device="mps")
    with torch.no_grad():
        out = dev.to_numpy(model(x))
    assert out.shape == (1, 1, 64, 64)
    state = dev.cpu_state_dict(model)
    assert {v.device.type for v in state.values()} == {"cpu"}
    assert {p.device.type for p in model.parameters()} == {"mps"}  # the model itself did not move


def test_cpu_state_dict_matches_the_model():
    model = ReconstructionUNet(base_features=16)
    state = dev.cpu_state_dict(model)
    assert list(state) == list(model.state_dict())
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    ReconstructionUNet(base_features=16).load_state_dict(state)


def test_scripts_never_convert_a_model_output_without_moving_it_to_the_host():
    offenders = []
    for path in glob.glob(os.path.join(ROOT, "scripts", "*.py")):
        for n, line in enumerate(open(path), 1):
            if re.search(r"\.numpy\(\)", line) and "to_numpy" not in line:
                offenders.append(f"{os.path.basename(path)}:{n}: {line.strip()}")
    assert offenders == []


def test_cuda_memory_is_only_reported_on_cuda(monkeypatch):
    assert dev.cuda_peak_memory(torch.device("cpu")) is None
    assert dev.cuda_peak_memory(torch.device("mps")) is None
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 456)
    assert dev.cuda_peak_memory(torch.device("cuda")) == {"peak_allocated_bytes": 123, "peak_reserved_bytes": 456}


def test_device_and_jax_metadata(monkeypatch):
    meta = dev.device_metadata("cpu", torch.device("cpu"))
    for key in ("requested_device", "resolved_device", "gpu_name", "python_version", "torch_version",
                "torch_cuda_version", "cudnn_version"):
        assert key in meta
    assert meta["resolved_device"] == "cpu" and meta["gpu_name"] is None
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    dev.enforce_jax_cpu()
    jax_meta = dev.assert_jax_cpu()
    assert jax_meta["jax_cpu_only"] is True and jax_meta["jax_platforms_env"] == "cpu"
    assert all("cpu" in d.lower() for d in jax_meta["jax_devices"])
