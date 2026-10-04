"""Tests for scripts/train.py and har.train.runner (synthetic data; real data where marked)."""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest
import torch
import yaml

from har.train.runner import UNKNOWN_USER, make_run_id, run_by
from har.train.trainer import HISTORY_FIELDS
from har.utils.config import ProtocolError
from har.utils.paths import CONFIGS, REPO_ROOT

# BUILD_SPEC 8.2: every run folder file, and every metrics.json key.
RUN_FILES = {
    "config.resolved.yaml", "metrics.json", "history.csv", "curves.png",
    "confusion_val.png", "confusion_test.png",
    "classification_report_val.txt", "classification_report_test.txt", "best.pt",
}
TEST_FILES = {"confusion_test.png", "classification_report_test.txt"}
SCHEMA_KEYS = {
    "model", "run_id", "git_sha", "git_dirty", "run_by", "status", "representation",
    "params_trainable", "model_size_mb", "best_epoch", "epochs_run", "train_time_s",
    "mean_epoch_time_s", "val", "test", "latency", "hardware", "val_subjects", "protocol_override",
}
SPLIT_KEYS = {"accuracy", "macro_f1", "per_class", "confusion"}

TINY_CONFIG = """\
name: tiny
data: {representation: synthetic, input_shape: [16]}
model: {target: torch.nn.Linear, params: {in_features: 16, out_features: 6}}
"""


def _load_script():
    spec = importlib.util.spec_from_file_location("train_script", REPO_ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_script = _load_script()


def _config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(TINY_CONFIG + extra, encoding="utf-8")
    return path


def _metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def test_smoke_run_writes_every_file_and_schema_key(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_dir = train_script.main(["--config", str(_config(tmp_path)), "--smoke", "--output-root", str(out)])

    assert run_dir.parent == out / "_smoke" / "tiny"
    assert {p.name for p in run_dir.iterdir()} == RUN_FILES

    m = _metrics(run_dir)
    assert SCHEMA_KEYS <= set(m)
    assert set(m["val"]) == SPLIT_KEYS and set(m["test"]) == SPLIT_KEYS
    assert m["model"] == "tiny" and m["run_id"] == run_dir.name
    assert m["representation"] == "synthetic" and m["protocol_override"] is False
    assert m["params_trainable"] == 16 * 6 + 6
    assert m["smoke"] is True and "smoke" in m["status"]
    assert 1 <= m["best_epoch"] <= m["epochs_run"] <= 2
    assert set(m["latency"]) == {"cpu_bs1", "device_bs1"}            # smoke: batch size 1 only
    assert m["latency"]["cpu_bs1"]["threads"] == 1 and m["latency"]["cpu_bs1"]["iters"] == 5
    assert m["run_id"].split("_")[1] == m["git_sha"]

    with (run_dir / "history.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == HISTORY_FIELDS and len(rows) == m["epochs_run"]

    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["train"]["max_epochs"] == 2 and resolved["run"]["smoke"] is True
    assert resolved["train"]["lr"] == 0.001               # the rest of the protocol comes from base.yaml

    state = torch.load(run_dir / "best.pt", weights_only=True)
    torch.nn.Linear(16, 6).load_state_dict(state)


def test_no_test_skips_the_test_split(tmp_path: Path) -> None:
    run_dir = train_script.main(["--config", str(_config(tmp_path)), "--smoke", "--no-test",
                                 "--output-root", str(tmp_path)])
    assert {p.name for p in run_dir.iterdir()} == RUN_FILES - TEST_FILES
    assert _metrics(run_dir)["test"] is None


def test_protocol_keys_need_the_override_flag_and_are_recorded(tmp_path: Path) -> None:
    config = _config(tmp_path, "seed: 7\n")
    with pytest.raises(ProtocolError):
        train_script.main(["--config", str(config), "--smoke", "--output-root", str(tmp_path)])
    run_dir = train_script.main(["--config", str(config), "--smoke", "--allow-protocol-override",
                                 "--output-root", str(tmp_path)])
    assert _metrics(run_dir)["protocol_override"] is True


def test_tag_and_repeated_runs_get_distinct_folders(tmp_path: Path) -> None:
    args = ["--config", str(_config(tmp_path)), "--smoke", "--tag", "lr check", "--output-root", str(tmp_path)]
    first, second = train_script.main(args), train_script.main(args)
    assert first != second and first.parent == second.parent
    assert "_lr-check" in first.name and _metrics(second)["run_id"] == second.name


def test_unknown_representation_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(TINY_CONFIG.replace("synthetic", "spectrogram"), encoding="utf-8")
    with pytest.raises(ValueError, match="representation"):
        train_script.main(["--config", str(config), "--smoke", "--output-root", str(tmp_path)])


def test_run_id_format() -> None:
    now = datetime(2026, 10, 3, 9, 15, 0)
    assert make_run_id("ab12cd3", now=now) == "20261003-091500_ab12cd3"
    assert make_run_id("ab12cd3", tag="try 2/b", now=now) == "20261003-091500_ab12cd3_try-2-b"


def test_run_by_falls_back_with_a_warning_when_git_cannot_answer(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="user.name"):
        assert run_by(tmp_path / "missing") == UNKNOWN_USER


@pytest.mark.needs_data
def test_smoke_run_on_raw_windows(tmp_path: Path) -> None:
    from har.data.splits import load_val_subjects

    run_dir = train_script.main(["--config", str(CONFIGS / "transformer.yaml"), "--smoke",
                                 "--output-root", str(tmp_path)])
    m = _metrics(run_dir)
    assert m["representation"] == "raw"
    assert m["counts"] == {"train": 512, "val": 2076, "test": 2947}
    assert m["val_subjects"] == load_val_subjects()
    assert sum(m["test"]["per_class"][c]["support"] for c in m["test"]["per_class"]) == 2947


@pytest.mark.needs_data
def test_smoke_run_on_engineered_features(tmp_path: Path) -> None:
    pytest.importorskip("har.data.features", reason="M1.1 feature pipeline not merged yet")
    run_dir = train_script.main(["--config", str(CONFIGS / "mlp.yaml"), "--smoke",
                                 "--output-root", str(tmp_path)])
    m = _metrics(run_dir)
    assert m["representation"] == "features"
    assert m["counts"]["val"] == 2076 and m["counts"]["test"] == 2947
