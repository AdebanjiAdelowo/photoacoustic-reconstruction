"""The intervention's training set and its pre-registered success criteria (no training, no simulation)."""
import numpy as np
import pytest

import scripts.edge_gap_intervention as E
import scripts.mixed_phantom_experiment as M
from src.image_statistics import image_statistics
from src.shape_phantoms import blur_edges, shape_phantom


def test_blur_edges():
    disc = shape_phantom("discs", 64, 1)
    assert blur_edges(disc, 0.0) is disc
    soft = blur_edges(disc, 1.0)
    assert soft.dtype == np.float32 and soft.max() == pytest.approx(1.0) and soft.min() >= 0.0
    sharp_g, soft_g = image_statistics(disc)["max_gradient"], image_statistics(soft)["max_gradient"]
    assert 0.25 < soft_g < 0.45 < sharp_g  # between the smooth blobs (about 0.2) and one-pixel edges (0.5 to 0.71)


def test_only_edge_sharpness_differs_from_the_control_training_set():
    control, edgefill = M.mixed_split(M.TRAIN_SEED_OFFSET, 160), E.edgefill_split(M.TRAIN_SEED_OFFSET, 160)
    changed = edgefill["blur"] > 0
    assert changed.sum() == 60 and len(edgefill["phantom"]) == len(control["phantom"]) == 160  # replaced, not added
    for name in ("seed", "n_sensors", "family"):
        assert np.array_equal(control[name], edgefill[name])
    assert np.array_equal(control["phantom"][~changed], edgefill["phantom"][~changed])
    for fam in E.SHARP_FAMILIES:  # half of every sharp family, balanced over the sensor counts
        sel = changed & (edgefill["family"] == fam)
        assert sel.sum() == 20 and (edgefill["n_sensors"][sel] == 16).sum() == 10
    assert not changed[edgefill["family"] == "blobs"].any()
    assert E.BLUR_RANGE[0] <= edgefill["blur"][changed].min() and edgefill["blur"][changed].max() <= E.BLUR_RANGE[1]
    for i in np.nonzero(changed)[0][:5]:  # the blurred copy is the same shape in the same place
        assert np.array_equal(edgefill["phantom"][i], blur_edges(control["phantom"][i], edgefill["blur"][i]))


def _rows(valley_new, sharp_loss=0.5, smooth_loss=0.2, thin_gain=2.0):
    """Synthetic results: control advantage 13, 5, 2, 7, 8, 8 dB over the blur sweep."""
    control = dict(zip((0.0, 0.5, 1.0, 2.0, 3.0, 4.0), (13.0, 5.0, 2.0, 7.0, 8.0, 8.0)))
    sweeps, shape = [], []
    for k in (16, 64):
        for b, c in control.items():
            new = valley_new if b in E.VALLEY_BLURS else c - (sharp_loss if b == 0 else smooth_loss)
            sweeps.append({"sweep": "disc_blur", "value": b, "sensors": k, "label": "high",
                           E.CONTROL: {"minus_tikhonov": [c, c - 0.3, c + 0.3]},
                           E.INTERVENTION: {"minus_tikhonov": [new, new - 0.3, new + 0.3]},
                           "intervention_minus_control": [new - c, new - c - 0.3, new - c + 0.3]})
        for w in (1.0, 1.5):
            sweeps.append({"sweep": "vessel_width", "value": w, "sensors": k, "label": "high",
                           E.CONTROL: {"minus_tikhonov": [-5.0, -5.3, -4.7]},
                           E.INTERVENTION: {"minus_tikhonov": [-5.0 + thin_gain, 0, 0]},
                           "intervention_minus_control": [thin_gain, thin_gain - 0.3, thin_gain + 0.3]})
        for fam in E.SHARP_FAMILIES:
            shape.append({"family": fam, "label": "high", "sensors": k, "intervention_minus_control": [-sharp_loss, 0, 0]})
    return sweeps, shape


def test_criteria_supported():
    c = E.evaluate_criteria(*_rows(valley_new=7.0))
    assert c["verdict"] == "supported" and not c["edge_gap_explanation_refuted"]
    v = c["by_sensors"]["16"]
    assert v["valley_gain"] == pytest.approx(5.0) and v["S1_valley"] and v["S2_no_harm"] and v["S3_transfer"]


def test_criteria_harm_and_failure():
    assert E.evaluate_criteria(*_rows(valley_new=7.0, sharp_loss=4.0))["verdict"] == "partly supported"
    assert E.evaluate_criteria(*_rows(valley_new=7.0, smooth_loss=1.5))["verdict"] == "partly supported"
    weak = E.evaluate_criteria(*_rows(valley_new=4.0, thin_gain=0.2))  # gain 2 dB: below the 3 dB criterion
    assert weak["verdict"] == "not supported" and not weak["edge_gap_explanation_refuted"]
    assert not weak["by_sensors"]["64"]["S3_transfer"]
    none = E.evaluate_criteria(*_rows(valley_new=2.5))  # gain 0.5 dB: the explanation is refuted
    assert none["verdict"] == "not supported" and none["edge_gap_explanation_refuted"]
