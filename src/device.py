"""Device selection, determinism and device metadata for the PyTorch U-Net.

Only the U-Net runs on an accelerator. The j-Wave/JAX physics (forward simulation, time reversal,
Tikhonov matrix assembly) stays on CPU: moving it would change the numerics of the data every method
is scored on. `enforce_jax_cpu` must therefore be called before JAX is imported in any workflow that
runs on a machine with a GPU.

Bitwise agreement between CPU, MPS and CUDA is not expected (different convolution kernels and
summation order). Runs on one device with the settings of `configure_determinism` are repeatable.
"""

import os
import platform
import sys
from collections import OrderedDict

import torch

DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


class DeviceUnavailableError(RuntimeError):
    """An explicitly requested accelerator is not available. There is no silent fallback."""


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def mps_available() -> bool:
    return bool(torch.backends.mps.is_available())


def resolve_device(choice: str = "auto") -> torch.device:
    """Map a --device choice to a torch.device.

    auto: CUDA, then MPS, then CPU. An explicit `cuda` or `mps` either resolves to that device or
    raises DeviceUnavailableError; it never falls back.
    """
    if choice not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {choice!r}; choose one of {', '.join(DEVICE_CHOICES)}")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not cuda_available():
            raise DeviceUnavailableError(
                "--device cuda was requested but torch.cuda.is_available() is False "
                f"(torch {torch.__version__}, built with CUDA: {torch.version.cuda}). "
                "Not falling back to another device.")
        return torch.device("cuda")
    if choice == "mps":
        if not mps_available():
            raise DeviceUnavailableError(
                "--device mps was requested but torch.backends.mps.is_available() is False. "
                "Not falling back to another device.")
        return torch.device("mps")
    if cuda_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_determinism(device: torch.device) -> dict:
    """Apply the reproducibility settings for `device` and return what was set.

    cpu:  torch.use_deterministic_algorithms(True), as the CPU runs of this project always did.
    cuda: the same, plus cuDNN deterministic, cuDNN benchmark off, TF32 off and
          CUBLAS_WORKSPACE_CONFIG (required by deterministic cuBLAS; it has to be in the environment
          before the first CUDA call, so it is only filled in when unset).
    mps:  nothing; the MPS backend has no deterministic mode and MPS runs were never repeatable.
    """
    settings = {"device_type": device.type, "deterministic_algorithms": False}
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True)
        settings["deterministic_algorithms"] = True
    elif device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        settings.update({
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        })
    return settings


def synchronize(device) -> None:
    """Wait for queued accelerator work, so that a wall-clock timer measures finished computation."""
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def to_numpy(tensor: torch.Tensor):
    """Tensor on any device -> numpy array on the host, values unchanged."""
    return tensor.detach().cpu().numpy()


def cpu_state_dict(model: torch.nn.Module) -> OrderedDict:
    """The model's state dict with every tensor on CPU, so a checkpoint carries no device tag."""
    state = model.state_dict()
    out = OrderedDict((k, v.detach().cpu()) for k, v in state.items())
    if hasattr(state, "_metadata"):
        out._metadata = state._metadata
    return out


def reset_cuda_peak_memory(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_memory(device):
    """Peak CUDA allocator figures in bytes since the last reset, or None off CUDA.

    These are PyTorch CUDA allocator statistics. They are not comparable with process RSS or with
    MPS memory figures.
    """
    if torch.device(device).type != "cuda":
        return None
    return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}


def device_metadata(requested: str, device: torch.device) -> dict:
    is_cuda = device.type == "cuda"
    return {
        "requested_device": requested,
        "resolved_device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if is_cuda else None,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if is_cuda else None,
        "cuda_available": cuda_available(),
        "mps_available": mps_available(),
    }


def enforce_jax_cpu() -> None:
    """Pin JAX to CPU. Call before the first `import jax` (directly or through j-Wave)."""
    already_imported = "jax" in sys.modules and os.environ.get("JAX_PLATFORMS") != "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    if already_imported:
        # too late for the variable to take effect: accept only if JAX is on CPU anyway
        assert_jax_cpu()


def jax_metadata() -> dict:
    import jax

    devices = [str(d) for d in jax.devices()]
    platforms = sorted({d.platform for d in jax.devices()})
    return {
        "jax_version": jax.__version__,
        "jax_devices": devices,
        "jax_default_backend": jax.default_backend(),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_cpu_only": platforms == ["cpu"] and jax.default_backend() == "cpu",
    }


def assert_jax_cpu() -> dict:
    meta = jax_metadata()
    if not meta["jax_cpu_only"]:
        raise RuntimeError(f"JAX is not CPU-only (devices {meta['jax_devices']}). The physics path "
                           "must stay on CPU: set JAX_PLATFORMS=cpu before starting Python.")
    return meta
