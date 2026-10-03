"""Protection of the historical outputs, isolated run directories, provenance and checkpoint
portability. Everything is written under pytest's temporary directory."""
import json
import os

import numpy as np
import pytest
import torch

from src import run_safety as rs
from src.reconstruction_net import ReconstructionUNet

ROOT = str(rs.REPO_ROOT)
CUDA = torch.device("cuda")
CPU = torch.device("cpu")


def test_historical_locations_are_recognised(tmp_path):
    for rel in ("report/mvp_results.txt", "report/new.json", "data/train.npz",
                "experiments/unet_checkpoint.pt", "experiments/expanded/unet_seed0.pt"):
        assert rs.is_historical(os.path.join(ROOT, rel)), rel
    assert not rs.is_historical(os.path.join(ROOT, "results", "run", "report", "mvp_results.txt"))
    assert not rs.is_historical(tmp_path / "report" / "x.txt")


def test_cuda_and_remote_runs_may_not_write_historical_outputs(monkeypatch, unprotected):
    target = os.path.join(ROOT, "report", "mvp_results.txt")
    assert rs.ensure_writable(target, device=CPU) is not None  # local behaviour is unchanged
    assert rs.ensure_writable(target, device=torch.device("mps")) is not None
    with pytest.raises(rs.ProtectedOutputError):
        rs.ensure_writable(target, device=CUDA)
    with pytest.raises(rs.ProtectedOutputError):
        rs.ensure_writable(os.path.join(ROOT, "data", "train.npz"), device="cuda:0")
    assert rs.ensure_writable(target, device=CUDA, allow_historical=True) is not None
    monkeypatch.setenv(rs.PROTECT_ENV, "1")
    with pytest.raises(rs.ProtectedOutputError):
        rs.ensure_writable(target, device=CPU)


def test_existing_files_outside_the_historical_locations_are_not_overwritten(tmp_path):
    target = tmp_path / "out.json"
    assert rs.ensure_writable(target) == target
    target.write_text("{}")
    with pytest.raises(rs.OutputExistsError):
        rs.ensure_writable(target)
    assert rs.ensure_writable(target, overwrite=True) == target


def test_historical_mps_checkpoint_is_protected(tmp_path, unprotected):
    with pytest.raises(rs.ProtectedOutputError):
        rs.ensure_new_checkpoint(rs.HISTORICAL_MPS_CHECKPOINT, device=CUDA)
    with pytest.raises(rs.ProtectedOutputError):
        rs.ensure_new_checkpoint(rs.HISTORICAL_MPS_CHECKPOINT, device=CUDA, overwrite=True)
    existing = tmp_path / "unet.pt"
    existing.write_bytes(b"trained network")
    with pytest.raises(rs.OutputExistsError):
        rs.ensure_new_checkpoint(existing, device=CPU)


def test_training_refuses_to_replace_a_checkpoint_before_doing_anything(tmp_path):
    from scripts.train import train

    existing = tmp_path / "unet.pt"
    existing.write_bytes(b"trained network")
    with pytest.raises(rs.OutputExistsError):
        train(CPU, checkpoint_path=str(existing), data_dir=str(tmp_path / "no-such-data"))
    assert existing.read_bytes() == b"trained network"


def test_evaluation_scripts_refuse_historical_outputs_on_cuda(monkeypatch, unprotected):
    import scripts.evaluate_mvp as mvp
    import scripts.evaluate_noise_sensitivity as noise
    import scripts.expanded_evaluation as expanded

    monkeypatch.setattr(mvp, "resolve_device", lambda choice: CUDA)
    monkeypatch.setattr(noise, "resolve_device", lambda choice: CUDA)
    monkeypatch.setattr(expanded, "get_device", lambda choice: CUDA)
    for run in (lambda: mvp.main("cuda"), lambda: noise.main("cuda"), lambda: expanded.main("cuda")):
        with pytest.raises(rs.ProtectedOutputError):
            run()


def test_run_directories(tmp_path):
    assert rs.run_dir("20260101T000000Z-abc1234-colab") == rs.REPO_ROOT / "results" / "20260101T000000Z-abc1234-colab"
    for bad in ("", "..", "a/b", "a b"):
        with pytest.raises(ValueError):
            rs.run_dir(bad)
    dirs = rs.make_run_dirs(tmp_path / "run")
    assert sorted(dirs) == ["checkpoints", "data", "report", "smoke"] and all(d.is_dir() for d in dirs.values())
    with pytest.raises(rs.ProtectedOutputError):
        rs.make_run_dirs(rs.REPO_ROOT / "report" / "run")


def test_results_directory_is_ignored_by_git():
    lines = [l.strip() for l in open(os.path.join(ROOT, ".gitignore"))]
    assert "results/" in lines


@pytest.fixture
def tiny_run(tmp_path):
    """A two-epoch training run on synthetic 16 x 16 data, entirely inside tmp_path."""
    from scripts.train import train

    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    for name, n in (("train", 8), ("val", 4)):
        np.savez(data / f"{name}.npz", phantom=rng.random((n, 16, 16), dtype=np.float32),
                 recon=rng.random((n, 16, 16), dtype=np.float32),
                 n_sensors=np.full(n, 16, np.int32), seed=np.arange(n, dtype=np.int64))
    ckpt, meta = tmp_path / "checkpoints" / "unet_seed3.pt", tmp_path / "checkpoints" / "unet_seed3.meta.json"
    history = train(CPU, epochs=2, seed=3, checkpoint_path=str(ckpt), verbose=False, data_dir=str(data),
                    metadata_path=str(meta), requested_device="cpu")
    return tmp_path, ckpt, meta, history


def test_training_provenance_fields(tiny_run):
    _, _, meta_path, history = tiny_run
    meta = json.loads(meta_path.read_text())
    for key in ("git_commit", "requested_device", "resolved_device", "gpu_name", "python_version",
                "torch_version", "torch_cuda_version", "cudnn_version", "jax_version", "jax_devices",
                "jax_cpu_only", "seed", "model", "training", "dataset", "timing", "cuda_memory", "determinism"):
        assert key in meta, key
    assert meta["seed"] == 3 and meta["resolved_device"] == "cpu" and meta["jax_cpu_only"] is True
    assert meta["model"]["parameter_count"] == 481745
    assert meta["training"] == {"epochs": 2, "batch_size": 8, "lr": 1e-3, "optimizer": "Adam", "loss": "MSELoss",
                                "checkpoint_selection": "lowest validation loss", "n_train": 8, "n_val": 4}
    assert set(meta["dataset"]) == {"train.npz", "val.npz"} and len(meta["dataset"]["train.npz"]["sha256"]) == 64
    assert meta["determinism"]["amp"] is False and meta["determinism"]["channels_last"] is False
    assert meta["cuda_memory"] is None  # CPU run: no CUDA allocator figures
    assert meta["timing"]["total_seconds"] > 0 and len(history["timing"]["epoch_seconds"]) == 2


def test_run_writes_only_inside_its_own_directory(tiny_run):
    tmp_path, ckpt, meta, _ = tiny_run
    written = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert written == ["checkpoints/unet_seed3.meta.json", "checkpoints/unet_seed3.pt", "data/train.npz", "data/val.npz"]


def test_checkpoint_is_device_neutral_and_keeps_its_format(tiny_run):
    _, ckpt, _, _ = tiny_run
    stored = torch.load(ckpt, weights_only=True)  # no map_location needed
    assert set(stored) == {"model_state", "epoch", "val_loss", "seed"}
    assert {v.device.type for v in stored["model_state"].values()} == {"cpu"}
    model = ReconstructionUNet(base_features=16)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model_state"])
    with torch.no_grad():
        assert torch.isfinite(model.eval()(torch.zeros(1, 1, 16, 16))).all()


def test_historical_checkpoints_still_load():
    path = rs.HISTORICAL_MPS_CHECKPOINT
    if not path.exists():
        pytest.skip("the original checkpoint is not present on this machine")
    stored = torch.load(path, map_location="cpu", weights_only=True)
    ReconstructionUNet(base_features=16).load_state_dict(stored["model_state"])
