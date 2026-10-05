"""Write colab/run_cuda.ipynb and kaggle/run_cuda.ipynb from one description.

The two notebooks are thin launchers around the same repository scripts. They differ only in the
working directory, where results are kept and how the archive is handed over. Every cell is plain
Python (no shell magics), and the notebooks are stored without outputs.

    python scripts/build_remote_notebooks.py          # rewrite both notebooks
    python scripts/build_remote_notebooks.py --check  # exit 1 if either file is out of date
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The commit the notebooks check out. It must contain the remote workflow (scripts/cuda_smoke.py and
# the files it uses). A commit cannot name itself, so REF names the commit that added the workflow;
# the commit that records this pin changes only this file and the two generated notebooks. To run a
# later version, set REF to its full SHA, rebuild the notebooks and commit them the same way.
REF = "7b5712295577f548c0d1f2070195eb52e794b4d0"
REPO_URL = "https://github.com/AdebanjiAdelowo/photoacoustic-reconstruction.git"

PLATFORMS = {
    "colab": {
        "title": "Photoacoustic reconstruction: CUDA smoke on Google Colab",
        "setup": ("Before running: Runtime > Change runtime type > select a GPU. Then Runtime > Run all.\n"
                  "The notebook records whichever GPU it is given."),
        "work_dir": "/content",
        "results_root": 'REPO_DIR / "results"',
        "message": "CUDA SMOKE COMPLETE. No full experiment was run.",
        "handover": ('try:\n'
                     '    from google.colab import files\n'
                     '    files.download(archive)\n'
                     'except Exception as exc:  # not in Colab, or the browser blocked the download\n'
                     '    print(f"download it from the file browser instead: {archive} ({exc})")'),
    },
    "kaggle": {
        "title": "Photoacoustic reconstruction: CUDA smoke on Kaggle",
        "setup": ("Before running, in the notebook settings (right-hand panel):\n"
                  "- Accelerator: choose a GPU. Any CUDA GPU works; the notebook records the one it is given.\n"
                  "- Internet: on. It is needed to clone the repository and install the pinned packages.\n\n"
                  "Then Run All. Everything is written under `/kaggle/working`, which Kaggle keeps as the "
                  "notebook output."),
        "work_dir": "/kaggle/working",
        "results_root": 'WORK_DIR / "results"',
        "message": "KAGGLE CUDA SMOKE COMPLETE. No full experiment was run.",
        "handover": 'print(f"archive kept in the notebook output: {archive}")',
    },
}

INTRO = """# {title}

{setup}

What this notebook does, in order: inspect the GPU, check out a pinned commit, install the pinned
dependencies without replacing the platform's CUDA PyTorch, run the tests, run the CUDA smoke
(`scripts/cuda_smoke.py`), validate its verdict, estimate the full-run time and package the results.
With `RUN_FULL = False` it stops there.

Only the PyTorch U-Net uses the GPU. The j-Wave/JAX physics is pinned to CPU (`JAX_PLATFORMS=cpu`).
All scientific code lives in the repository; this notebook only launches it. See `REMOTE_GPU.md`.
"""

CONFIG = '''import os
import pathlib
import time

REPO_URL = "{repo_url}"
# REF is the full SHA of the commit that is checked out and run. It must contain the remote
# workflow (scripts/cuda_smoke.py); change it in scripts/build_remote_notebooks.py, not here.
REF = "{ref}"
RUN_FULL = False  # True runs the gated five-seed CPU-versus-CUDA experiment after the smoke

PLATFORM = "{platform}"
WORK_DIR = pathlib.Path("{work_dir}")
REPO_DIR = WORK_DIR / "photoacoustic-reconstruction"
RESULTS_ROOT = {results_root}

os.environ["JAX_PLATFORMS"] = "cpu"                       # the physics stays on CPU
os.environ["PHOTOACOUSTIC_PROTECT_HISTORICAL"] = "1"      # no writes to report/, data/, experiments/
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"         # deterministic cuBLAS

# One RUN_ID per session: created once, kept in a file, reused when this cell is run again.
_run_id_file = WORK_DIR / "photoacoustic_run_id.txt"
if _run_id_file.exists():
    RUN_ID = _run_id_file.read_text().strip()
else:
    RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + REF[:7] + "-" + PLATFORM
    _run_id_file.write_text(RUN_ID)
RUN_DIR = RESULTS_ROOT / RUN_ID
FULL_GATE_OPEN = RUN_FULL is True
print("RUN_ID", RUN_ID)
print("RUN_DIR", RUN_DIR)
print("RUN_FULL", RUN_FULL)
'''

HELPERS = '''import json
import shutil
import subprocess
import sys


def sh(cmd, cwd=None):
    """Run a command, stream its output here and stop on failure."""
    print("$", " ".join(str(c) for c in cmd))
    proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"command failed with exit status {proc.returncode}: {cmd[0]}")


def py(*args):
    """Run a repository script with this interpreter, from the repository root."""
    sh([sys.executable, *args], cwd=REPO_DIR)


print(sys.version)
sh(["nvidia-smi"])
'''

CHECKOUT = '''if not (REPO_DIR / ".git").exists():
    sh(["git", "clone", REPO_URL, REPO_DIR])
sh(["git", "fetch", "--all", "--tags"], cwd=REPO_DIR)
sh(["git", "checkout", "--detach", REF], cwd=REPO_DIR)

head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout.strip()
assert head == REF, f"HEAD is {head}, expected {REF}"
dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout
assert dirty == "", "source tree is not clean:\\n" + dirty
print("HEAD", head, "(clean)")
'''

INSTALL = '''ENV_DIR = RUN_DIR / "env"
py("scripts/remote_setup.py", "preflight", "--out", ENV_DIR / "preflight.json")
py("scripts/remote_setup.py", "install", "--preflight", ENV_DIR / "preflight.json")
py("scripts/remote_setup.py", "verify", "--preflight", ENV_DIR / "preflight.json", "--out", ENV_DIR / "environment.json")
'''

TESTS = '''py("-m", "pytest", "-q", "-p", "no:cacheprovider")
'''

SMOKE = '''SMOKE_DIR = RUN_DIR / "smoke"
VERDICT = SMOKE_DIR / "smoke_verdict.json"
if VERDICT.exists():
    print("smoke verdict already present for this RUN_ID; not running it again")
else:
    py("scripts/cuda_smoke.py", "--device", "cuda", "--out-dir", SMOKE_DIR)
'''

VALIDATE = '''py("scripts/cuda_smoke.py", "--validate", VERDICT)

verdict = json.loads(VERDICT.read_text())
assert verdict["passed"] is True and verdict["require_cuda"] is True
assert verdict["provenance"]["resolved_device"].startswith("cuda")
assert verdict["provenance"]["jax_cpu_only"] is True
print("GPU:", verdict["provenance"]["gpu_name"])
'''

ESTIMATE = '''py("scripts/remote_experiment.py", "--run-dir", RUN_DIR, "--estimate", VERDICT)
'''

PACKAGE_SMOKE = '''archive = shutil.make_archive(str(WORK_DIR / f"photoacoustic_{{RUN_ID}}_smoke"), "zip", root_dir=RESULTS_ROOT, base_dir=RUN_ID)
print("packaged", archive)
{handover}
'''

STOP = '''if not FULL_GATE_OPEN:
    print("{message}")
'''

FULL = '''if FULL_GATE_OPEN:
    py("scripts/remote_experiment.py", "--run-dir", RUN_DIR, "--confirm-full")
'''

PACKAGE_FULL = '''if FULL_GATE_OPEN:
    archive = shutil.make_archive(str(WORK_DIR / f"photoacoustic_{RUN_ID}_full"), "zip", root_dir=RESULTS_ROOT, base_dir=RUN_ID)
    print("packaged", archive)
    print("FULL EXPERIMENT COMPLETE.")
'''

GATE_NOTE = """## Full experiment (gated)

The cells below do nothing unless `RUN_FULL = True` was set in the configuration cell. With the gate
open they run `scripts/remote_experiment.py`: data generated once on CPU, then the U-Net trained and
evaluated for seeds 0 to 4 on CPU (arm A) and on CUDA (arm B), and the two arms compared. The Tikhonov
and model-mismatch evaluations are not included.
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text, role, shared=True):
    return {"cell_type": "code", "execution_count": None, "outputs": [],
            "metadata": {"role": role, "shared": shared}, "source": text.splitlines(keepends=True)}


def build(platform):
    cfg = PLATFORMS[platform]
    cells = [
        md(INTRO.format(title=cfg["title"], setup=cfg["setup"])),
        md("## 1. Configuration"),
        code(CONFIG.format(repo_url=REPO_URL, ref=REF, platform=platform, work_dir=cfg["work_dir"],
                           results_root=cfg["results_root"]), "config", shared=False),
        md("## 2. GPU and interpreter"),
        code(HELPERS, "helpers"),
        md("## 3. Repository at the pinned commit"),
        code(CHECKOUT, "checkout"),
        md("## 4. Dependencies\n\nThe platform's CUDA PyTorch is kept; JAX is the CPU build."),
        code(INSTALL, "install"),
        md("## 5. Tests"),
        code(TESTS, "tests"),
        md("## 6. CUDA smoke"),
        code(SMOKE, "smoke"),
        md("## 7. Verdict"),
        code(VALIDATE, "validate"),
        md("## 8. Estimated full-run time\n\nAn estimate from the components measured by the smoke, not a measurement."),
        code(ESTIMATE, "estimate"),
        md("## 9. Package the smoke results"),
        code(PACKAGE_SMOKE.format(handover=cfg["handover"]), "package_smoke", shared=False),
        md("## 10. Stop"),
        code(STOP.format(message=cfg["message"]), "stop", shared=False),
        md(GATE_NOTE),
        code(FULL, "full"),
        code(PACKAGE_FULL, "package_full"),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


def render(platform):
    nb = build(platform)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"{platform}-{i:02d}"
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def path_for(platform):
    return os.path.join(ROOT, platform, "run_cuda.ipynb")


def main(argv=None):
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    stale = []
    for platform in PLATFORMS:
        text, path = render(platform), path_for(platform)
        if check:
            if not os.path.exists(path) or open(path).read() != text:
                stale.append(path)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        print("wrote", os.path.relpath(path, ROOT))
    if stale:
        print("out of date:", ", ".join(os.path.relpath(p, ROOT) for p in stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
