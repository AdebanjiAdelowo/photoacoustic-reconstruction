"""Output protection, isolated run directories and provenance.

The committed `report/` files, the local checkpoints under `experiments/` and the generated data under
`data/` are the evidence behind the published results. One of them, `experiments/unet_checkpoint.pt`
(trained on MPS), cannot be regenerated exactly. Remote and CUDA runs therefore write under
`results/<RUN_ID>/` and are refused access to the historical locations.

Rules applied by `ensure_writable`:
  * a historical location is refused when protection is active (the environment variable
    PHOTOACOUSTIC_PROTECT_HISTORICAL=1, or a CUDA device), unless explicitly allowed;
  * anywhere else, an existing file is refused unless overwriting is explicitly requested.
Local CPU/MPS runs with the default paths keep their historical behaviour.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_DIRS = ("report", "data", "experiments")
HISTORICAL_MPS_CHECKPOINT = REPO_ROOT / "experiments" / "unet_checkpoint.pt"
PROTECT_ENV = "PHOTOACOUSTIC_PROTECT_HISTORICAL"
RUN_SUBDIRS = ("data", "checkpoints", "report", "smoke")


class ProtectedOutputError(RuntimeError):
    """A write to a historical location was refused."""


class OutputExistsError(RuntimeError):
    """A write would replace an existing file and overwriting was not requested."""


def is_historical(path) -> bool:
    """True if `path` lies in the repository's report/, data/ or experiments/ directory."""
    p = Path(path).resolve()
    return any(p == (REPO_ROOT / d).resolve() or (REPO_ROOT / d).resolve() in p.parents
               for d in HISTORICAL_DIRS)


def protection_active(device=None) -> bool:
    if os.environ.get(PROTECT_ENV) == "1":
        return True
    kind = getattr(device, "type", device)
    return kind is not None and str(kind).startswith("cuda")


def ensure_writable(path, *, device=None, overwrite: bool = False, allow_historical: bool = False) -> Path:
    """Check that `path` may be written, and return it. Raises instead of overwriting."""
    p = Path(path)
    if is_historical(p):
        if protection_active(device) and not allow_historical:
            raise ProtectedOutputError(
                f"refusing to write {p}: it is a historical output location and this is a "
                f"remote/CUDA run. Write under results/<RUN_ID>/ instead (--out-dir, --data-dir, "
                f"--checkpoint), or pass --allow-historical-overwrite if this is intentional.")
        return p
    if p.exists() and not overwrite:
        raise OutputExistsError(f"refusing to overwrite existing {p}; pass --overwrite or use a new run directory.")
    return p


def ensure_new_checkpoint(path, *, device=None, overwrite: bool = False, allow_historical: bool = False) -> Path:
    """Checkpoint paths are stricter: an existing checkpoint is never replaced implicitly, wherever
    it is, because a trained network may not be reproducible (the MPS original is not)."""
    p = Path(path)
    if is_historical(p) and protection_active(device) and not allow_historical:
        raise ProtectedOutputError(
            f"refusing to write checkpoint {p}: historical location in a remote/CUDA run. "
            "Use --checkpoint results/<RUN_ID>/checkpoints/....")
    if p.exists() and not overwrite:
        raise OutputExistsError(
            f"refusing to overwrite existing checkpoint {p}; pass --overwrite if it may be replaced, "
            "or choose another --checkpoint path.")
    return p


def run_dir(run_id: str, root=None) -> Path:
    """results/<run_id>/ (under `root`, default the repository's results/)."""
    if not run_id or any(c in run_id for c in "/\\ ") or run_id in (".", ".."):
        raise ValueError(f"invalid run id {run_id!r}")
    return (Path(root) if root is not None else REPO_ROOT / "results") / run_id


def make_run_dirs(base) -> dict:
    base = Path(base)
    if is_historical(base):
        raise ProtectedOutputError(f"run directory {base} is inside a historical location")
    dirs = {name: base / name for name in RUN_SUBDIRS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dataset_fingerprint(paths) -> dict:
    """sha256 and array shapes of each .npz split, keyed by file name."""
    import numpy as np

    out = {}
    for path in paths:
        path = Path(path)
        with np.load(path) as d:
            arrays = {k: [list(d[k].shape), str(d[k].dtype)] for k in d.files}
        out[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "arrays": arrays}
    return out


def git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None}


def write_json(path, payload, *, device=None, overwrite: bool = False) -> Path:
    p = ensure_writable(path, device=device, overwrite=overwrite)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return p
