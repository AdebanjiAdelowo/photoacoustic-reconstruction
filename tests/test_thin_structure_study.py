"""The thin-structure study before its evaluation: locked design, test set, plumbing and verdict logic.

Nothing here reconstructs the pre-registered test set: the plumbing runs on a small grid with a
stand-in forward model and stand-in reconstructions, and the verdict logic on invented scores.
"""
import copy
import inspect
import json
import os
import subprocess

import numpy as np
import pytest

import scripts.thin_structure_study as T
from src.run_safety import sha256_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = T.load_config()


def small_config(n_images=4, grid_size=16):
    config = copy.deepcopy(CONFIG)
    config["phantom"]["grid_size"], config["test_set"]["n_images"] = grid_size, n_images
    return config


def stand_in(config):
    """An identity 'forward model' and reconstructions that differ by a fixed offset per method."""
    names = [T.TIKHONOV] + T.network_names(config, "control") + T.network_names(config, "intervention")
    seen = {}

    def record(phantoms, k):
        seen.setdefault(k, []).append(phantoms.copy())
        return [p.reshape(-1).copy() for p in phantoms]

    def reconstruct(Y, k):
        size = config["phantom"]["grid_size"]
        return {n: np.stack(Y).reshape(-1, size, size) + 0.001 * i for i, n in enumerate(names)}
    return record, reconstruct, seen


# --------------------------------------------------------------------------- the locked design

def test_grid_is_the_full_factorial_without_the_empty_cell():
    cells = T.cell_list(CONFIG)
    assert len(cells) == 2 * (5 * 5 - 1) == len(set(cells)) and ("straight", 0, 0) not in cells
    profiles = {(w, s) for _, w, s in cells}
    assert all((g, w, s) in cells for g in CONFIG["phantom"]["geometries"] for w, s in profiles)
    c = CONFIG["cells"]
    named = [c["thin"], c["thick_same_edge"], c["sharp_thin_bar"], *c["thin_soft_set"], *c["sharp_familiar_bars"]]
    named += [pair[side] for pair in c["secondary_width_contrasts"] for side in ("thick", "thin")]
    named += [[c["edge_sweep_core_width"], s] for s in c["valley_edge_sigmas"] + c["edge_range_sigmas"] + [c["smooth_edge_sigma"]]]
    assert all(tuple(cell) in profiles for cell in named)                 # every cell a hypothesis uses is evaluated
    assert {1, 1.5, 2} <= {s for w, s in profiles if w == 0}              # the widths of the earlier vessel sweep


def test_primary_regime_is_14_db_at_both_sensor_counts():
    e = CONFIG["evaluation"]
    assert e["noise_label"] == "high" and e["relative_noise_std"] == 0.2 and e["sensor_counts"] == [16, 64]
    assert 20 * np.log10(1 / e["relative_noise_std"]) == pytest.approx(14.0, abs=0.05)
    assert e["groups"] == {"control": "mixed160", "intervention": "edgefill160"} and e["network_seeds"] == [0, 1, 2, 3, 4]
    with open(os.path.join(ROOT, "report/mixed_phantom_results.json")) as f:
        selected = json.load(f)["tikhonov_weights_selected_on_mixed160"]
    assert all(selected[f"high|{k}"] == e["tikhonov_mu"][str(k)] for k in e["sensor_counts"])


def test_test_seeds_are_new_and_fixed():
    import scripts.mixed_phantom_experiment as M
    import scripts.prior_shift_analysis as P
    from scripts.expanded_evaluation import N_PER_SENSOR, SENSOR_COUNTS, TEST_SEED_OFFSET
    from scripts.generate_training_data import SPLIT_SEED_OFFSETS, SPLITS

    seeds = set(T.image_seeds(CONFIG))
    assert len(seeds) == CONFIG["test_set"]["n_images"] == 50 and min(seeds) == 600_000
    used = set(range(TEST_SEED_OFFSET, TEST_SEED_OFFSET + N_PER_SENSOR * len(SENSOR_COUNTS)))
    for name, n in SPLITS.items():
        used |= set(range(SPLIT_SEED_OFFSETS[name], SPLIT_SEED_OFFSETS[name] + n))
    for offset, n in ((M.TRAIN_SEED_OFFSET, 160), (M.VAL_SEED_OFFSET, 32), (M.TEST_SEED_OFFSET, 5000),
                      (P.VESSEL_SEED_OFFSET, P.N_SWEEP), (P.DISC_SEED_OFFSET, P.N_SWEEP)):
        used |= set(range(offset, offset + n))
    assert not seeds & used
    assert CONFIG["evaluation"]["noise_seed"] not in (12345, 555_000, 777_000, 778_000, 868_000, 969_000)


def test_phantoms_are_deterministic_and_shared_by_every_cell():
    a, b = (T.test_phantoms(CONFIG, "curved", 2, 1) for _ in range(2))
    assert a.shape == (50, 64, 64) and np.array_equal(a, b)
    mask = T.long_line_mask(CONFIG)
    assert mask.shape == (50,) and 25 <= mask.sum() < 50      # a robustness subset, not the whole set or a sliver


def test_preregistration_document_states_the_locked_numbers():
    with open(os.path.join(ROOT, CONFIG["preregistration"])) as f:
        text = f.read()
    t, s, e = CONFIG["thresholds_db"], CONFIG["test_set"], CONFIG["evaluation"]
    for needle in (str(s["seed_offset"]), str(e["noise_seed"]), f"{s['n_images']} images", "14 dB",
                   f"{t['material_effect']:g} dB", f"{t['negligible_effect']:g} dB", f"{t['interaction_flag']:g} dB",
                   "configs/thin_structure_study.json", "committed before"):
        assert needle in text, needle
    assert all(os.path.exists(os.path.join(ROOT, p)) for p in CONFIG["locked_files"])


# --------------------------------------------------------------------------- plumbing

def test_noise_is_reproducible_and_independent_of_cell_and_order():
    clean_a, clean_b = np.linspace(1, 2, 400, dtype=np.float32), np.linspace(-3, 5, 400, dtype=np.float32)
    def draw(clean, i=3, k=64):
        return (T.add_noise(clean, 0.2, 979_000, i, k) - clean) / np.sqrt(np.mean(clean ** 2))
    assert np.array_equal(T.add_noise(clean_a, 0.2, 979_000, 3, 64), T.add_noise(clean_a, 0.2, 979_000, 3, 64))
    assert np.allclose(draw(clean_a), draw(clean_b), atol=1e-6)            # the same draw for one image in every cell
    assert not np.allclose(draw(clean_a), draw(clean_a, i=4)) and not np.allclose(draw(clean_a), draw(clean_a, k=16))
    big = np.ones(200_000, np.float32)
    assert np.std(T.add_noise(big, 0.2, 1, 0, 16) - big) == pytest.approx(0.2, rel=0.02)
    assert T.add_noise(clean_a, 0.2, 1, 0, 16).dtype == np.float32


def test_both_sensor_counts_see_identical_phantoms():
    config = small_config()
    record, reconstruct, seen = stand_in(config)
    T.evaluate(config, record, reconstruct)
    assert set(seen) == {16, 64} and len(seen[16]) == len(T.cell_list(config))
    assert all(np.array_equal(a, b) for a, b in zip(seen[16], seen[64]))


def test_evaluation_is_reproducible_and_complete():
    config = small_config()
    runs = [T.evaluate(config, *stand_in(config)[:2]) for _ in range(2)]
    assert runs[0].keys() == runs[1].keys() and all(np.array_equal(runs[0][k], runs[1][k]) for k in runs[0])
    n_methods = 1 + 2 * len(config["evaluation"]["network_seeds"])
    assert len(runs[0]) == len(T.cell_list(config)) * 2 * n_methods * 2          # cells x sensor counts x methods x metrics
    assert all(v.shape == (4,) and np.all(np.isfinite(v)) for v in runs[0].values())
    record, reconstruct, _ = stand_in(config)
    with pytest.raises(ValueError):                                                # a missing method is an error, not a gap
        T.evaluate(config, record, lambda Y, k: {T.TIKHONOV: reconstruct(Y, k)[T.TIKHONOV]})


def test_result_schema_records_every_factor_and_both_absolute_and_relative_quality():
    config = small_config()
    per_image = T.evaluate(config, *stand_in(config)[:2])
    rows = T.summarise(per_image, config)
    assert len(rows) == len(T.cell_list(config)) * 2
    row = rows[0]
    assert {"geometry", "core_width_px", "edge_sigma_px", "sensors", "noise_label", "n_images", "grid_size", "profile",
            "measured", "tikhonov", "control", "intervention", "intervention_minus_control"} <= set(row)
    assert set(row["profile"]) == {"fwhm", "plateau_width", "max_slope"} and set(row["tikhonov"]) == {"psnr", "ssim"}
    for role in ("control", "intervention"):
        r = row[role]
        assert {"group", "psnr", "ssim", "psnr_per_network", "ssim_per_network", "psnr_network_sd", "ssim_network_sd",
                "psnr_minus_tikhonov", "psnr_minus_tikhonov_per_network", "psnr_minus_tikhonov_network_sd"} <= set(r)
        assert len(r["psnr_per_network"]) == 5 and len(r["psnr"]) == 3
        assert np.mean(r["psnr_per_network"]) == pytest.approx(r["psnr"][0])
    json.dumps({"cells": rows, "hypotheses": T.evaluate_hypotheses(per_image, config)})   # serialisable


def test_report_files_can_be_written_from_the_results(tmp_path):
    config = small_config()
    per_image = T.evaluate(config, *stand_in(config)[:2])
    results = {"preregistration": config["preregistration"], "design_commit": "0" * 40,
               "cells": T.summarise(per_image, config), "hypotheses": T.evaluate_hypotheses(per_image, config)}
    T.write_text(results, tmp_path / "results.txt")
    T.plot(results, config, tmp_path / "results.png")
    text = (tmp_path / "results.txt").read_text()
    assert text.count(" sensors: Tikhonov ") == len(T.cell_list(config)) * 2 and "H4 sharp thin bar" in text
    order = [text.index(h) for h in ("== 1. Primary pre-registered contrasts", "== 2. Interactions",
                                     "== 3. Complete response surface", "== 4. Summary classification")]
    assert order == sorted(order)                                # the pre-registered order of evidence
    assert (tmp_path / "results.png").stat().st_size > 10_000


def test_script_cannot_train_or_overwrite():
    source = inspect.getsource(T)
    for forbidden in (".train(", "optim", ".backward(", "torch.save", "overwrite=True", "edgefill_split", "mixed_split"):
        assert forbidden not in source, forbidden
    assert "require_locked(config)" in inspect.getsource(T.main)


def test_generating_and_scoring_the_test_set_writes_nothing(tmp_path, monkeypatch):
    def snapshot():
        out = {}
        for d in ("data", "experiments", "report"):
            for folder, _, files in os.walk(os.path.join(ROOT, d)):
                for name in files:
                    path = os.path.join(folder, name)
                    out[path] = (os.path.getsize(path), os.path.getmtime(path))
        return out
    before = snapshot()
    monkeypatch.chdir(tmp_path)
    config = small_config()
    T.summarise(T.evaluate(config, *stand_in(config)[:2]), config)
    T.test_phantoms(CONFIG, "straight", 1, 0)
    assert snapshot() == before and os.listdir(tmp_path) == []


def test_existing_checkpoints_are_only_read():
    from scripts.checkpoint_manifest import CKPT_DIR, MANIFEST_PATH, verify
    path = os.path.join(ROOT, CKPT_DIR, "edgefill160_seed0.pt")
    if not os.path.exists(path):
        pytest.skip("checkpoints are gitignored and not present")
    import scripts.unrolled_evaluation as U
    before = (sha256_file(path), os.path.getmtime(path))
    model, _ = U.load_model(path, "cpu")
    assert not model.training and (sha256_file(path), os.path.getmtime(path)) == before
    with open(os.path.join(ROOT, MANIFEST_PATH)) as f:
        assert verify(json.load(f), os.path.join(ROOT, CKPT_DIR)) == []


def test_evaluation_refuses_an_uncommitted_or_modified_design(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=tmp_path, check=True, capture_output=True)
    config = {"locked_files": ["design.json", "runner.py"]}
    git("init")
    for name in config["locked_files"]:
        (tmp_path / name).write_text("locked\n")
    with pytest.raises(T.NotLockedError, match="not committed"):
        T.require_locked(config, tmp_path)
    git("add", "."), git("commit", "-m", "design")
    assert len(T.require_locked(config, tmp_path)) == 40
    (tmp_path / "design.json").write_text("changed after the commit\n")
    with pytest.raises(T.NotLockedError, match="differs from the commit"):
        T.require_locked(config, tmp_path)


# --------------------------------------------------------------------------- verdict logic on invented scores

def test_every_width_contrast_holds_the_edge_width_fixed():
    c = CONFIG["cells"]
    pairs = [{"thick": c["thick_same_edge"], "thin": c["thin"]}, *c["secondary_width_contrasts"]]
    assert len(pairs) == 3
    for pair in pairs:
        assert pair["thick"][1] == pair["thin"][1] and pair["thick"][0] - pair["thin"][0] >= 5


def scores(advantage, config=CONFIG, n=12):
    """per_image scores with Tikhonov at 30 dB and both groups at 30 + advantage(geometry, w, s, k, role)."""
    wobble = np.linspace(-0.2, 0.2, n)
    per_image = {}
    for g, w, s in T.cell_list(config):
        for k in config["evaluation"]["sensor_counts"]:
            key = f"{T.cell_key(g, w, s)}__{k}"
            per_image[f"{key}__{T.TIKHONOV}__psnr"] = np.full(n, 30.0)
            for role in ("control", "intervention"):
                for j, name in enumerate(T.network_names(config, role)):
                    per_image[f"{key}__{name}__psnr"] = 30.0 + advantage(g, w, s, k, role) + wobble + 0.1 * (j - 2)
    return per_image


def base(g, w, s, k, role, thin_penalty=11.0, curve_penalty=0.0, valley=True, bar_penalty=11.0):
    """Thick structures 7 dB ahead; the control dips at s = 0.5 and 1; thin structures pay a penalty."""
    a = 7.0
    if valley and role == "control" and s in (0.5, 1) and w >= 4:
        a -= 5.0
    if s > 0 and w <= 1:
        a -= thin_penalty if s <= 1 else thin_penalty / 2
    if s == 0 and w <= 1:
        a -= bar_penalty
    if g == "curved" and w <= 1:
        a -= curve_penalty
    return a


def verdict(**kw):
    return T.evaluate_hypotheses(scores(lambda *a: base(*a, **kw)), CONFIG)["64"]


def test_evidence_is_ordered_and_the_classification_comes_last():
    assert list(verdict()) == ["P0_failure_present", "H1_thinness", "H2_edge_profile", "H3_geometry", "H4_sharp_thin_bar",
                               "interactions", "edge_range_at_thick_core", "summary_classification"]


def test_thinness_dominated_pattern():
    h = verdict()
    assert h["P0_failure_present"]["holds"] and h["H2_edge_profile"]["status"] == "supported"
    for g in ("straight", "curved"):
        h1 = h["H1_thinness"][g]
        assert h1["status"] == "supported" and h1["primary_thick_minus_thin"]["estimate"] == pytest.approx(11.0)
        assert len(h1["primary_thick_minus_thin"]["per_network"]) == 5
        assert h1["secondary_fixed_edge"]["w6_s0.5_minus_w1_s0.5"]["estimate"] == pytest.approx(11.0)
        assert h1["secondary_fixed_edge"]["w6_s1.5_minus_w0_s1.5"]["estimate"] == pytest.approx(5.5)
    assert h["H3_geometry"]["status"] == T.GEOMETRY_NEGLIGIBLE and h["H4_sharp_thin_bar"]["status"] == "supported"
    assert not any(e["flagged"] for e in h["interactions"].values())
    assert h["summary_classification"] == "thinness-dominated"


def test_h1_verdict_uses_only_the_fixed_edge_primary_contrast():
    def adv(g, w, s, k, role):   # every other edge width is made useless as evidence
        return base(g, w, s, k, role) if s == 1 else -20.0
    h1 = T.evaluate_hypotheses(scores(adv), CONFIG)["64"]["H1_thinness"]["straight"]
    assert h1["status"] == "supported" and h1["primary_thick_minus_thin"]["estimate"] == pytest.approx(11.0)
    assert all(e["status"] == "rejected" for e in h1["secondary_fixed_edge"].values())
    assert verdict(thin_penalty=2.0)["H1_thinness"]["straight"]["status"] == "indeterminate"


def test_a_flagged_interaction_withholds_the_dominant_label():
    h = verdict(bar_penalty=6.0)   # thin soft lines lose 11 dB, thin sharp bars 6: the cost of thinness depends on the edge
    assert h["H4_sharp_thin_bar"]["status"] == "supported" and h["interactions"]["width_x_edge"]["flagged"]
    assert h["interactions"]["width_x_edge"]["estimate"] == pytest.approx(5.0)
    assert h["summary_classification"] == "thinness supported, dominance not established (interaction flagged: width_x_edge)"


def test_thresholds_are_applied_as_written():
    assert verdict(bar_penalty=2.9)["H4_sharp_thin_bar"]["status"] == "indeterminate"
    assert verdict(bar_penalty=0.5)["H4_sharp_thin_bar"]["status"] == "rejected"
    assert verdict(bar_penalty=0.5)["summary_classification"].startswith("thinness supported, dominance not established")


def test_geometry_contrast_is_signed_straight_minus_curved():
    h = verdict(curve_penalty=0.5)
    assert h["H3_geometry"]["straight_minus_curved"]["w0_s1"]["estimate"] == pytest.approx(0.5)
    assert h["H3_geometry"]["status"] == T.GEOMETRY_NEGLIGIBLE
    modest = verdict(curve_penalty=1.5)
    assert modest["H3_geometry"]["status"] == T.GEOMETRY_MODEST
    assert modest["summary_classification"].startswith("thinness supported, dominance not established")
    worse = verdict(curve_penalty=3.5)
    assert worse["H3_geometry"]["status"] == T.CURVED_WORSE
    assert worse["summary_classification"].startswith("mixed: thinness, geometry")


def test_opposite_direction_is_not_evidence_for_a_curvature_cause():
    h = verdict(curve_penalty=-3.5)   # curved thin lines 3.5 dB better than straight ones
    assert h["P0_failure_present"]["holds"] and h["H3_geometry"]["status"] == T.CURVED_BETTER
    assert h["H3_geometry"]["straight_minus_curved"]["w0_s1"]["estimate"] == pytest.approx(-3.5)
    label = h["summary_classification"]
    assert "geometry" not in label.split("(")[0] and "not evidence that curvature causes the failure" in label
    def adv(g, w, s, k, role):        # width explains little, and curved thin lines are the better ones
        return (-0.6 if g == "curved" else -3.8) if (w <= 1 and s > 0) else -1.0
    reversed_sign = T.evaluate_hypotheses(scores(adv), CONFIG)["64"]
    assert reversed_sign["P0_failure_present"]["holds"] and reversed_sign["H3_geometry"]["status"] == T.CURVED_BETTER
    assert reversed_sign["H1_thinness"]["straight"]["status"] == "indeterminate"
    assert reversed_sign["summary_classification"].startswith("inconclusive")


def test_geometry_dominated_pattern():
    def adv(g, w, s, k, role):   # straight thin lines are fine; curved thin lines fail
        return base(g, w, s, k, role, thin_penalty=0.0, curve_penalty=10.0, bar_penalty=0.0)
    h = T.evaluate_hypotheses(scores(adv), CONFIG)["64"]
    assert h["P0_failure_present"]["holds"] and h["H1_thinness"]["straight"]["status"] == "rejected"
    assert h["H3_geometry"]["status"] == T.CURVED_WORSE
    assert h["interactions"]["width_x_geometry"]["flagged"] and not h["interactions"]["width_x_edge"]["flagged"]
    assert h["interactions"]["width_x_geometry"]["estimate"] == pytest.approx(10.0)
    assert h["summary_classification"] == "geometry-dominated (interaction flagged: width_x_geometry)"


def test_edge_profile_dominated_pattern():
    def adv(g, w, s, k, role):   # the deficit follows the edge width at every core width
        return {0: 7.0, 0.5: 2.0, 1: -4.0, 1.5: 2.0, 2: 7.0}[s] + (5.0 if role == "intervention" and s in (0.5, 1) else 0.0) - 5.0 * (s == 1)
    h = T.evaluate_hypotheses(scores(adv), CONFIG)["64"]
    assert h["P0_failure_present"]["holds"] and h["H1_thinness"]["straight"]["status"] == "rejected"
    assert h["H1_thinness"]["straight"]["primary_thick_minus_thin"]["estimate"] == pytest.approx(0.0)
    assert h["edge_range_at_thick_core"] >= 3.0 and h["summary_classification"] == "edge-profile-dominated"


def test_no_classification_without_the_failure_and_flag_without_the_positive_control():
    assert verdict(thin_penalty=5.0, bar_penalty=5.0)["summary_classification"] == "failure not reproduced: no classification"
    h = verdict(valley=False)
    assert h["H2_edge_profile"]["status"] == "not reproduced"
    assert h["summary_classification"].endswith("(edge axis not validated: H2 not reproduced)")


def test_contrast_separates_network_and_tikhonov_changes_and_respects_the_subset():
    per_image = scores(lambda *a: base(*a))
    per_image[f"{T.cell_key('straight', 6, 1)}__64__{T.TIKHONOV}__psnr"] -= 4.0   # Tikhonov alone gets worse on the thick cell
    e = T.contrast(per_image, CONFIG, 64, "intervention", [("straight", 6, 1)], [("straight", 0, 1)])
    assert e["estimate"] == pytest.approx(15.0) and e["tikhonov_psnr_change"] == pytest.approx(-4.0)
    assert e["network_psnr_change"] == pytest.approx(11.0) and e["ci"][0] <= e["estimate"] <= e["ci"][1]
    mask = np.arange(12) < 6
    sub = T.evaluate_hypotheses(per_image, CONFIG, mask)["64"]
    assert sub["H1_thinness"]["straight"]["primary_thick_minus_thin"]["estimate"] == pytest.approx(15.0)
