"""Protocol checks: disjoint data splits, training-only calibration and regularisation selection.

The first group needs no data. The second compares committed report files with the (gitignored)
training data and is skipped when that data has not been generated.
"""
import json
import os

import numpy as np
import pytest

from scripts.expanded_evaluation import N_PER_SENSOR, SENSOR_COUNTS, TEST_SEED_OFFSET
from scripts.generate_training_data import SPLIT_SEED_OFFSETS, SPLITS
from src.calibration import fit_scales_per_sensor_count

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seed_ranges():
    ranges = {name: set(range(SPLIT_SEED_OFFSETS[name], SPLIT_SEED_OFFSETS[name] + n)) for name, n in SPLITS.items()}
    ranges["test_expanded"] = set(range(TEST_SEED_OFFSET, TEST_SEED_OFFSET + N_PER_SENSOR * len(SENSOR_COUNTS)))
    return ranges


def test_phantom_seed_ranges_of_all_splits_are_disjoint():
    ranges = list(_seed_ranges().values())
    for i, a in enumerate(ranges):
        for b in ranges[i + 1:]:
            assert not a & b


def test_expanded_test_set_has_equal_sensor_allocation():
    n = N_PER_SENSOR * len(SENSOR_COUNTS)
    alloc = np.array([SENSOR_COUNTS[i % 2] for i in range(n)])   # the rule in generate_test_set
    assert n == 400 and all((alloc == k).sum() == N_PER_SENSOR for k in SENSOR_COUNTS)


def test_regularisation_parameter_is_the_argmax_of_training_scores():
    sel = json.load(open(os.path.join(ROOT, "report/tikhonov_results.json")))["selection"]
    for key, s in sel.items():
        scores = s["train_psnr_by_mu"]
        best = max(scores, key=scores.get)
        assert float(best) == pytest.approx(s["selected_mu"]), key


needs_data = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data/train.npz")),
                                reason="training data not generated (data/*.npz is gitignored)")


@needs_data
def test_saved_calibration_gains_come_from_the_training_split():
    train = np.load(os.path.join(ROOT, "data/train.npz"))
    gains = fit_scales_per_sensor_count(train["recon"], train["phantom"], train["n_sensors"])
    saved = json.load(open(os.path.join(ROOT, "report/expanded_eval_results.json")))["calibration"]
    for k, c in saved.items():
        assert c["train_gain"] == pytest.approx(gains[int(k)], rel=1e-9)
        assert c["n_train_images"] == int((train["n_sensors"] == int(k)).sum())


@needs_data
def test_stored_splits_match_their_seed_ranges_and_do_not_overlap():
    ranges = _seed_ranges()
    seen = set()
    for name in ("train", "val", "test"):
        seeds = set(np.load(os.path.join(ROOT, f"data/{name}.npz"))["seed"].tolist())
        assert seeds == ranges[name]
        assert not seeds & seen
        seen |= seeds
    path = os.path.join(ROOT, "data/test_expanded.npz")
    if os.path.exists(path):
        d = np.load(path)
        assert set(d["seed"].tolist()) == ranges["test_expanded"] and not set(d["seed"].tolist()) & seen
        assert all((d["n_sensors"] == k).sum() == N_PER_SENSOR for k in SENSOR_COUNTS)
