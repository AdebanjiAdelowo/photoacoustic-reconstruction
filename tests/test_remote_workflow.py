"""The remote GPU workflow: launch notebooks, smoke script, full-run gate and device comparison.
Runs without CUDA; the smoke protocol is exercised on CPU and is then rejected as a CUDA smoke."""
import ast
import copy
import json
import os
import re
import time

import numpy as np
import pytest

import scripts.build_remote_notebooks as builder
import scripts.compare_devices as compare_devices
import scripts.cuda_smoke as smoke
import scripts.remote_experiment as remote
import scripts.remote_setup as remote_setup
from src import device as dev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATFORMS = ("colab", "kaggle")
MESSAGES = {"colab": "CUDA SMOKE COMPLETE. No full experiment was run.",
            "kaggle": "KAGGLE CUDA SMOKE COMPLETE. No full experiment was run."}


def notebook(platform):
    with open(os.path.join(ROOT, platform, "run_cuda.ipynb")) as f:
        return json.load(f)


def code_cells(platform):
    return [c for c in notebook(platform)["cells"] if c["cell_type"] == "code"]


def cell(platform, role):
    (found,) = [c for c in code_cells(platform) if c["metadata"]["role"] == role]
    return "".join(found["source"])


# --- notebooks ---

@pytest.mark.parametrize("platform", PLATFORMS)
def test_notebook_is_valid_and_stored_without_outputs(platform):
    nb = notebook(platform)
    assert nb["nbformat"] == 4 and nb["cells"]
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown") and isinstance(c["source"], list)
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse("".join(c["source"]))  # plain Python, no shell magics


def test_notebooks_match_the_builder():
    assert builder.main(["--check"]) == 0


@pytest.mark.parametrize("platform", PLATFORMS)
def test_configuration_defaults(platform):
    assigned = {t.id: node.value for node in ast.parse(cell(platform, "config")).body
                if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name)}
    assert isinstance(assigned["RUN_FULL"], ast.Constant) and assigned["RUN_FULL"].value is False
    assert re.fullmatch(r"[0-9a-f]{40}", assigned["REF"].value)
    assert assigned["PLATFORM"].value == platform
    src = cell(platform, "config")
    assert 'os.environ["JAX_PLATFORMS"] = "cpu"' in src
    assert 'os.environ["PHOTOACOUSTIC_PROTECT_HISTORICAL"] = "1"' in src
    assert ('"/kaggle/working"' in src) == (platform == "kaggle")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_run_id_survives_rerunning_the_configuration_cell(platform, tmp_path, monkeypatch):
    for var in ("JAX_PLATFORMS", "PHOTOACOUSTIC_PROTECT_HISTORICAL", "CUBLAS_WORKSPACE_CONFIG"):
        monkeypatch.setenv(var, os.environ.get(var, ""))  # restored after the test
    work_dir = builder.PLATFORMS[platform]["work_dir"]
    src = cell(platform, "config").replace(f'pathlib.Path("{work_dir}")', f"pathlib.Path({str(tmp_path)!r})")
    stamps = iter(["20260101T000000Z", "20260202T000000Z"])
    monkeypatch.setattr(time, "strftime", lambda *a: next(stamps))
    first, second = {}, {}
    exec(src, first)
    exec(src, second)
    assert first["RUN_ID"] == second["RUN_ID"] == f"20260101T000000Z-{builder.REF[:7]}-{platform}"
    assert first["FULL_GATE_OPEN"] is False
    assert str(first["RUN_DIR"]).endswith(os.path.join("results", first["RUN_ID"]))


@pytest.mark.parametrize("platform", PLATFORMS)
def test_smoke_stop_and_full_run_gate(platform):
    cells = code_cells(platform)
    roles = [c["metadata"]["role"] for c in cells]
    stop = roles.index("stop")
    assert roles[stop + 1:] == ["full", "package_full"]
    assert f'print("{MESSAGES[platform]}")' in cell(platform, "stop")
    for c in cells[stop + 1:]:  # everything after the stop is inside the gate
        body = ast.parse("".join(c["source"])).body
        assert len(body) == 1 and isinstance(body[0], ast.If)
        assert isinstance(body[0].test, ast.Name) and body[0].test.id == "FULL_GATE_OPEN" and not body[0].orelse
    for c in cells[:stop + 1]:  # nothing before it can start the experiment
        assert "--confirm-full" not in "".join(c["source"])
    assert "--confirm-full" in cell(platform, "full")


def test_colab_and_kaggle_run_the_same_protocol():
    shared = {p: [(c["metadata"]["role"], "".join(c["source"])) for c in code_cells(p) if c["metadata"]["shared"]]
              for p in PLATFORMS}
    assert shared["colab"] == shared["kaggle"]
    for p in PLATFORMS:
        differing = {c["metadata"]["role"] for c in code_cells(p) if not c["metadata"]["shared"]}
        assert differing == {"config", "package_smoke", "stop"}  # setup, paths, packaging only
        assert '"scripts/cuda_smoke.py", "--device", "cuda"' in cell(p, "smoke")
        source = "\n".join("".join(c["source"]) for c in code_cells(p))
        assert not re.search(r"^\s*(import|from)\s+(torch|jax|jwave|numpy)\b", source, re.M)  # thin launcher
        assert {n.name for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.ClassDef))} == {"sh", "py"}


# --- dependency strategy ---

def test_remote_install_keeps_platform_torch_and_cpu_jax():
    lines = remote_setup.remote_requirements()
    names = [re.split(r"[=<>]", l)[0].strip() for l in lines]
    assert "torch" not in names
    pinned = {name for name, spec, _ in remote_setup.parse_requirements() if spec}
    assert set(remote_setup.SCIENTIFIC_PINS) <= pinned and set(remote_setup.SCIENTIFIC_PINS) <= set(names)
    assert not any("cuda" in l.lower() for l in lines)  # no GPU build of JAX


def test_preinstalled_jax_gpu_plugins_are_removed(monkeypatch):
    installed = {"jax-cuda12-plugin": "0.11.1", "jax-cuda12-pjrt": "0.11.1"}
    commands = []
    monkeypatch.setattr(remote_setup, "installed_version", installed.get)
    monkeypatch.setattr(remote_setup.subprocess, "run",
                        lambda cmd: commands.append(cmd) or type("Result", (), {"returncode": 0})())
    assert remote_setup.remove_jax_gpu_plugins() == ["jax-cuda12-plugin", "jax-cuda12-pjrt"]
    assert commands[0][1:] == ["-m", "pip", "uninstall", "--yes", "jax-cuda12-plugin", "jax-cuda12-pjrt"]
    assert not any("torch" in part for part in commands[0][1:])
    installed.clear()
    assert remote_setup.remove_jax_gpu_plugins() == [] and len(commands) == 1


def test_compiled_code_is_released_periodically(monkeypatch):
    import src.jax_cache as jax_cache

    cleared = []
    monkeypatch.setattr(jax_cache.jax, "clear_caches", lambda: cleared.append(jax_cache._simulations))
    monkeypatch.setattr(jax_cache, "_simulations", 0)
    marker = object()
    assert all(jax_cache.simulation_done(marker) is marker for _ in range(2 * jax_cache.CLEAR_EVERY + 1))
    assert cleared == [jax_cache.CLEAR_EVERY, 2 * jax_cache.CLEAR_EVERY]
    # about 80 memory mappings per simulation must stay far below Linux's 65,530 per process
    assert 80 * jax_cache.CLEAR_EVERY < 65530 / 4


# --- full-run gate ---

def test_full_experiment_does_not_start_without_confirmation(tmp_path, capsys):
    run_dir = tmp_path / "results" / "run"
    assert remote.main(["--run-dir", str(run_dir)]) == 2
    assert remote.main(["--run-dir", str(run_dir), "--plan"]) == 0
    assert "FULL EXPERIMENT NOT STARTED" in capsys.readouterr().out
    assert not run_dir.exists() and list(tmp_path.iterdir()) == []


def test_full_experiment_plan(tmp_path):
    p = remote.plan(tmp_path / "run")
    assert p["arms"] == ["cpu", "cuda"] and p["seeds"] == [0, 1, 2, 3, 4]
    paths = [p["data_dir"], p["tr_cache"], p["comparison_dir"], *p["checkpoint_dirs"].values(),
             *p["report_dirs"].values(), *p["mvp_dirs"].values()]
    assert all(os.path.commonpath([path, p["run_dir"]]) == p["run_dir"] for path in paths)
    assert "tikhonov_evaluation.py" in p["excluded"] and "model_mismatch_evaluation.py" in p["excluded"]
    assert p["workload"] == {"forward_simulations": 856, "time_reversals": 2456,
                             "optimiser_steps_per_arm": 1500, "metric_images_per_arm": 14000}
    assert "NOT the original" in remote.MVP_NOTE


# --- smoke ---

def test_explicit_cuda_smoke_fails_cleanly_without_cuda(tmp_path, no_accelerators):
    out = tmp_path / "smoke"
    with pytest.raises(dev.DeviceUnavailableError):
        smoke.run_smoke("cuda", str(out))
    assert not out.exists()


@pytest.fixture(scope="module")
def cpu_smoke(tmp_path_factory):
    import torch

    was = torch.are_deterministic_algorithms_enabled()
    out = tmp_path_factory.mktemp("run") / "smoke"
    verdict = smoke.run_smoke("cpu", str(out), require_cuda=False)
    torch.use_deterministic_algorithms(was)
    return out, verdict


def test_smoke_protocol_runs_on_cpu(cpu_smoke):
    out, verdict = cpu_smoke
    assert verdict["passed"], verdict["failed_checks"]
    assert sorted(os.listdir(out)) == ["smoke_checkpoint.pt", "smoke_verdict.json"]
    assert json.loads((out / "smoke_verdict.json").read_text())["passed"] is True
    names = {c["name"] for c in verdict["checks"]}
    assert {"jax_cpu_only", "jax_cpu_only_after_simulation", "parameters_on_device", "batch_on_device",
            "losses_finite", "gradients_finite", "loss_decreased", "output_shape", "output_finite",
            "psnr_finite", "ssim_finite", "checkpoint_is_device_neutral", "cpu_reload_matches_device_output",
            "recording_shape_image0", "reconstruction_shape_image3"} <= names
    assert verdict["protocol"]["sensor_counts"] == [16, 16, 64, 64] and len(verdict["losses"]) == 20
    assert verdict["max_abs_difference_cpu_reload_vs_device"] <= smoke.OUTPUT_TOLERANCE
    assert verdict["cuda_memory"]["training"] is None
    with pytest.raises(Exception):  # a second run into the same directory is refused
        smoke.run_smoke("cpu", str(out), require_cuda=False)


def test_a_cpu_smoke_is_not_accepted_as_a_cuda_smoke(cpu_smoke):
    _, verdict = cpu_smoke
    assert smoke.validate_verdict(verdict, require_cuda=False) == []
    assert smoke.validate_verdict(verdict, require_cuda=True)

    as_cuda = copy.deepcopy(verdict)  # what a passing CUDA verdict looks like
    as_cuda["require_cuda"] = True
    as_cuda["provenance"]["resolved_device"] = "cuda"
    as_cuda["cuda_memory"]["training"] = {"peak_allocated_bytes": 1, "peak_reserved_bytes": 2}
    assert smoke.validate_verdict(as_cuda) == []
    for breakage in (lambda v: v.update(passed=False),
                     lambda v: v["checks"][0].update(passed=False),
                     lambda v: v["provenance"].update(jax_cpu_only=False),
                     lambda v: v.update(max_abs_difference_cpu_reload_vs_device=1e-2)):
        broken = copy.deepcopy(as_cuda)
        breakage(broken)
        assert smoke.validate_verdict(broken)


def test_runtime_estimate_from_smoke_timings(cpu_smoke):
    est = remote.estimate(cpu_smoke[1])
    assert est["total_seconds"] > 0 and 0 < est["physics_share"] < 1
    assert est["total_seconds"] == pytest.approx(est["physics_seconds"] + est["training_seconds_cpu_arm"]
                                                 + est["training_seconds_cuda_arm"] + est["metrics_seconds"])


# --- device comparison ---

def _arm(directory, shift, rng):
    os.makedirs(directory)
    n_sensors = np.array([16, 64] * 10, np.int32)
    arrays = {"n_sensors": n_sensors, "seed": 100000 + np.arange(20)}
    for metric, base in (("psnr", 30.0), ("ssim", 0.7)):
        arrays[f"noiseless__tr_raw__{metric}"] = np.full(20, base - 10)
        arrays[f"noiseless__tr_cal__{metric}"] = np.full(20, base - 4)
        seeds = [base + s * 0.1 + shift + rng.normal(0, 0.01, 20) for s in range(5)]
        for s, v in enumerate(seeds):
            arrays[f"noiseless__seed{s}__{metric}"] = v
        arrays[f"noiseless__unet_mean_over_seeds__{metric}"] = np.mean(seeds, axis=0)
    np.savez(os.path.join(directory, compare_devices.PER_IMAGE), **arrays)


def test_device_comparison(tmp_path):
    rng = np.random.default_rng(0)
    _arm(tmp_path / "cpu", 0.0, rng)
    _arm(tmp_path / "cuda", 0.05, rng)
    r = compare_devices.compare(str(tmp_path / "cpu"), str(tmp_path / "cuda"))
    assert r["time_reversal_identical_in_both_arms"] is True and r["networks"] == [f"seed{s}" for s in range(5)]
    e = r["levels"][0]["by_sensors"]["16"]["psnr"]
    assert len(e["cpu"]["per_seed_means"]) == 5 and e["cpu"]["sd_over_seeds"] > 0
    assert e["b_minus_a"]["mean_over_seeds_difference"]["mean"] == pytest.approx(0.05, abs=0.02)
    assert e["cuda"]["unet_minus_tr_cal"]["mean"] == pytest.approx(4.25, abs=0.05)
    assert e["published_context"]["published_mean"] == pytest.approx(31.84, abs=0.01)  # read-only context
    assert "cuda_mean_within_published_seed_range" in e["published_context"]
    compare_devices.write(r, str(tmp_path / "comparison"))
    text = (tmp_path / "comparison" / "device_comparison.txt").read_text()
    assert "cuda - cpu" in text and "statistical, not bitwise" in text
    with pytest.raises(Exception):
        compare_devices.write(r, str(tmp_path / "comparison"))
