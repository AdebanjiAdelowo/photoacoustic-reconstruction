"""Environment preparation for a remote GPU machine (Colab, Kaggle): no science happens here.

The platform's CUDA build of PyTorch is kept. Everything else is installed at the versions pinned in
requirements.txt, with pip constrained so that it cannot replace torch. JAX stays the CPU build from
PyPI, and a GPU plugin for JAX that the platform preinstalled is removed: it belongs to the platform's
own JAX version, the pinned jaxlib cannot load it, and the physics must not run on a GPU in any case.
If the pinned set cannot be installed on the platform's Python, the install step fails and the
conflict is reported; nothing is substituted silently.

    python scripts/remote_setup.py preflight --out results/<RUN_ID>/env/preflight.json
    python scripts/remote_setup.py install   --preflight results/<RUN_ID>/env/preflight.json
    python scripts/remote_setup.py verify    --preflight results/<RUN_ID>/env/preflight.json \
                                             --out results/<RUN_ID>/env/environment.json

`preflight` and `install` import neither JAX nor NumPy, so the freshly installed versions are the
ones the later steps see.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from importlib import metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")
# torch is taken from the platform (it carries the CUDA runtime); every other requirement is installed
PLATFORM_PROVIDED = ("torch",)
# packages whose installed version must equal the pin: they determine the numbers
SCIENTIFIC_PINS = ("numpy", "scipy", "scikit-image", "jax", "jaxlib", "jaxdf", "jwave")
JAX_GPU_DISTRIBUTIONS = ("jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda11-plugin", "jax-cuda11-pjrt")


def parse_requirements(path=REQUIREMENTS):
    """[(name, spec-or-None, original line)] for every requirement line."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==\s*(\S+))?$", line)
            if not m:
                raise ValueError(f"unsupported requirement line: {line!r}")
            out.append((m.group(1).lower(), m.group(3), line))
    return out


def remote_requirements(path=REQUIREMENTS):
    """Requirement lines to install remotely: everything except the platform-provided packages."""
    return [line for name, _, line in parse_requirements(path) if name not in PLATFORM_PROVIDED]


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def torch_status():
    import torch

    cuda = torch.cuda.is_available()
    return {"torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "cuda_available": cuda, "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_count": torch.cuda.device_count() if cuda else 0}


def preflight(out):
    status = {"python_version": sys.version.split()[0], "platform": platform.platform(),
              "executable": sys.executable, **torch_status(),
              "installed_before": {name: installed_version(name) for name, _, _ in parse_requirements()},
              "jax_gpu_distributions_before": {d: installed_version(d) for d in JAX_GPU_DISTRIBUTIONS},
              "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS")}
    print(json.dumps(status, indent=2))
    if not status["cuda_available"]:
        raise SystemExit("PREFLIGHT FAILED: torch.cuda.is_available() is False. Enable a GPU runtime "
                         "before installing anything.")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(status, f, indent=2)
    return status


def install(preflight_path):
    with open(preflight_path) as f:
        before = json.load(f)
    lines = remote_requirements()
    with tempfile.TemporaryDirectory() as tmp:
        req, con = os.path.join(tmp, "requirements-remote.txt"), os.path.join(tmp, "constraints.txt")
        with open(req, "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(con, "w") as f:  # pip may not move torch off the platform's CUDA build
            f.write(f"torch=={before['torch_version']}\n")
        cmd = [sys.executable, "-m", "pip", "install", "--no-input", "-r", req, "-c", con]
        print("installing:", ", ".join(lines))
        print("constraint: torch==" + before["torch_version"])
        result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            "INSTALL FAILED: the pinned requirements could not be installed on this platform "
            f"(Python {sys.version.split()[0]}). Do not loosen the pins to continue: the scientific "
            "dependencies define the numbers. Record the pip error above and see REMOTE_GPU.md.")
    remove_jax_gpu_plugins()


def remove_jax_gpu_plugins():
    """Uninstall preinstalled JAX GPU plugins. jaxlib imports a plugin whenever one is installed, even
    with JAX_PLATFORMS=cpu, and a plugin built for another JAX version makes `import jax` fail."""
    present = [d for d in JAX_GPU_DISTRIBUTIONS if installed_version(d)]
    if not present:
        print("no JAX GPU plugin installed; nothing to remove")
        return []
    print("removing JAX GPU plugins (JAX runs on CPU only):", ", ".join(present))
    if subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", *present]).returncode != 0:
        raise SystemExit("INSTALL FAILED: could not remove the JAX GPU plugins: " + ", ".join(present))
    return present


def verify(preflight_path, out):
    with open(preflight_path) as f:
        before = json.load(f)
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit("VERIFY FAILED: JAX_PLATFORMS must be 'cpu' before JAX is imported.")
    problems = []
    after = torch_status()
    if after["torch_version"] != before["torch_version"]:
        problems.append(f"torch changed from {before['torch_version']} to {after['torch_version']}")
    if not after["cuda_available"]:
        problems.append("torch.cuda.is_available() is False after installation")
    pins = {name: spec for name, spec, _ in parse_requirements()}
    installed = {name: installed_version(name) for name in pins}
    for name in SCIENTIFIC_PINS:
        if installed.get(name) != pins.get(name):
            problems.append(f"{name}: installed {installed.get(name)}, pinned {pins.get(name)}")

    gpu_plugins = {d: v for d in JAX_GPU_DISTRIBUTIONS if (v := installed_version(d))}
    if gpu_plugins:
        problems.append(f"JAX GPU plugins are still installed: {gpu_plugins}")

    sys.path.insert(0, ROOT)
    from src.device import jax_metadata

    try:
        jax_meta = jax_metadata()
    except Exception as exc:  # e.g. a plugin or NumPy that the pinned jaxlib cannot work with
        raise SystemExit(f"VERIFY FAILED: JAX could not be imported or initialised: {type(exc).__name__}: {exc}\n"
                         f"  other problems: {problems}")
    if not jax_meta["jax_cpu_only"]:
        problems.append(f"JAX is not CPU-only: {jax_meta['jax_devices']}")
    report = {
        "python_version": sys.version.split()[0], "platform": platform.platform(), **after, **jax_meta,
        "installed": installed, "pinned": pins,
        "deviations_from_requirements": {
            "torch": {"pinned": pins.get("torch"), "installed": after["torch_version"],
                      "reason": "the platform's CUDA build of PyTorch is kept; requirements.txt pins the "
                                "version used on the development machine"}}
        if after["torch_version"] != pins.get("torch") else {},
        "jax_gpu_distributions_present": gpu_plugins,
        "jax_gpu_distributions_removed": {d: v for d, v in before.get("jax_gpu_distributions_before", {}).items()
                                          if v and d not in gpu_plugins},
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    if problems:
        raise SystemExit("VERIFY FAILED:\n  " + "\n  ".join(problems))
    print("ENVIRONMENT VERIFIED: CUDA PyTorch preserved, JAX on CPU, scientific pins installed.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--out", required=True)
    p = sub.add_parser("install")
    p.add_argument("--preflight", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--preflight", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        preflight(args.out)
    elif args.command == "install":
        install(args.preflight)
    else:
        verify(args.preflight, args.out)


if __name__ == "__main__":
    main()
