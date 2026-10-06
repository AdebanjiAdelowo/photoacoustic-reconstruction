"""The checkpoint manifest: verification of a set of files, and consistency of the committed records."""
import json
import os
import re

import pytest

from scripts.checkpoint_manifest import ARCHIVE_RECORD_PATH, CKPT_DIR, GROUPS, MANIFEST_PATH, SEEDS, checkpoint_names, verify
from src.run_safety import sha256_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def test_verify_detects_missing_truncated_and_altered_files(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"weights-a")
    (tmp_path / "b.pt").write_bytes(b"weights-b")
    manifest = {"checkpoints": [{"filename": n, "bytes": 9, "sha256": sha256_file(tmp_path / n)} for n in ("a.pt", "b.pt")]}
    assert verify(manifest, tmp_path) == []
    (tmp_path / "a.pt").write_bytes(b"weights-A")                      # same size, other content
    (tmp_path / "b.pt").write_bytes(b"weights")                        # truncated
    assert verify(manifest, tmp_path) == ["a.pt: SHA-256 differs from the manifest", "b.pt: 7 bytes, expected 9"]
    (tmp_path / "b.pt").unlink()
    assert verify(manifest, tmp_path)[1] == "b.pt: missing"


def test_committed_manifest_covers_the_ten_networks_and_links_them_to_the_results():
    manifest = _load(MANIFEST_PATH)
    entries = manifest["checkpoints"]
    assert [e["filename"] for e in entries] == checkpoint_names() and len(entries) == 10
    assert sorted((e["group"], e["seed"]) for e in entries) == sorted((g, s) for g in GROUPS for s in SEEDS)
    for e in entries:
        assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) and e["bytes"] > 0 and e["n_parameters"] == 28365
        results = _load(e["link_to_committed_results"]["results_file"])["networks"][e["filename"][:-3]]
        assert e["stored_metadata"]["epoch"] == results["epoch"] and e["stored_metadata"]["val_loss"] == results["val_loss"]
        assert e["link_to_committed_results"]["stored_epoch_and_val_loss_equal_committed"]
        assert abs(e["link_to_committed_results"]["val_loss_recomputed_minus_stored"]) < 1e-9
    check = manifest["integrity_check"]
    assert check["group_mean_max_abs_difference_db"] < 1e-6 and check["control_per_network_max_abs_psnr_difference_db"] < 1e-3


def test_manifest_does_not_claim_an_unrecorded_training_environment():
    manifest = _load(MANIFEST_PATH)
    history = manifest["historical_training_environment"]
    assert history["recorded_at_training_time"] is False
    assert history["python"] == history["torch"] == history["other_packages"] == "unknown"
    assert "basis" in history["device"] and manifest["environment_at_verification"]["packages"]["torch"]


def test_archive_record_matches_the_manifest():
    record, manifest = _load(ARCHIVE_RECORD_PATH), _load(MANIFEST_PATH)
    assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) and record["archive_filename"].endswith(".tar.gz")
    members = {m["name"]: m for m in record["members"]}
    assert set(members) == {e["path"] for e in manifest["checkpoints"]} | {MANIFEST_PATH}
    for e in manifest["checkpoints"]:
        assert members[e["path"]]["sha256"] == e["sha256"] and members[e["path"]]["bytes"] == e["bytes"]
    assert members[MANIFEST_PATH]["sha256"] == sha256_file(os.path.join(ROOT, MANIFEST_PATH))


def test_local_checkpoints_match_the_manifest():
    directory = os.path.join(ROOT, CKPT_DIR)
    if not all(os.path.exists(os.path.join(directory, n)) for n in checkpoint_names()):
        pytest.skip("checkpoints are gitignored and not present")
    assert verify(_load(MANIFEST_PATH), directory) == []
