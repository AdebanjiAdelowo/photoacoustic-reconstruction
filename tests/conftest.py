import os

import pytest
import torch

from src.run_safety import PROTECT_ENV


@pytest.fixture
def restore_torch_flags():
    """Tests that switch on determinism settings must not leak them into other tests."""
    saved = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
             torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    yield
    torch.use_deterministic_algorithms(saved[0])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved[1], saved[2]
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[3], saved[4]
    if saved[5] is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved[5]


@pytest.fixture
def unprotected(monkeypatch):
    """Historical-output protection off, whatever the environment the tests run in."""
    monkeypatch.delenv(PROTECT_ENV, raising=False)


@pytest.fixture
def no_accelerators(monkeypatch):
    monkeypatch.setattr("src.device.cuda_available", lambda: False)
    monkeypatch.setattr("src.device.mps_available", lambda: False)
