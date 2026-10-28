"""Tests for har.eval.latency_check and scripts/gru_latency_check.py (PHASE3_SPEC F3.3)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import torch

from har.eval.latency_check import measure_onednn_effect, render_latency_check_md, summarise, write_latency_check
from har.utils.paths import REPO_ROOT

BASE_YAML = """\
seed: 42
final_seeds: [42, 43, 44]
data: {augment: false}
train: {batch_size: 64, max_epochs: 50, optimizer: adam, lr: 0.001, weight_decay: 0.0, patience: 10,
        monitor: val_macro_f1, mode: max, grad_clip_norm: 1.0, device: auto}
eval: {latency_batch_sizes: [1], latency_warmup: 1, latency_iters: 3, latency_threads: 1}
"""
HW = {"torch": "9.9", "cpu": "Test CPU", "os": "Test OS"}


def _rnn(name: str, cell: str, pooling: str = "last") -> str:
    return (f"name: {name}\nowner: M3\ndata: {{representation: raw}}\nmodel:\n  target: har.models.birnn.BiRNN\n"
            f"  params: {{in_channels: 9, hidden: 4, num_layers: 2, cell: {cell}, bidirectional: true,\n"
            f"           dropout: 0.3, pooling: {pooling}, n_classes: 6}}\n")


def _configs(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    (configs / "final").mkdir(parents=True)
    (configs / "base.yaml").write_text(BASE_YAML, encoding="utf-8")
    (configs / "bilstm.yaml").write_text(_rnn("bilstm", "lstm"), encoding="utf-8")
    (configs / "gru.yaml").write_text(_rnn("gru", "gru"), encoding="utf-8")
    (configs / "final" / "gru.yaml").write_text(_rnn("gru", "gru", "mean"), encoding="utf-8")
    return configs


def _row(model: str, onednn: bool, repeat: int, ms: float) -> dict:
    return {"model": model, "config": f"configs/final/{model}.yaml", "params": 100, "onednn": onednn,
            "repeat": repeat, "mean_ms": ms, "p50_ms": ms, "p95_ms": ms, "threads": 1, "warmup": 20, "iters": 200}


def test_measures_both_models_in_both_conditions_and_restores_the_flag(tmp_path: Path) -> None:
    before = torch.backends.mkldnn.enabled
    rows = measure_onednn_effect(configs_dir=_configs(tmp_path), repeats=2)
    assert torch.backends.mkldnn.enabled == before
    assert len(rows) == 8
    assert sorted({(r["model"], r["onednn"]) for r in rows}) == [("bilstm", False), ("bilstm", True),
                                                                 ("gru", False), ("gru", True)]
    assert all(r["mean_ms"] > 0 and r["threads"] == 1 and r["iters"] == 3 for r in rows)
    configs = {r["model"]: r["config"] for r in rows}
    assert configs["gru"].endswith("final/gru.yaml") and configs["bilstm"].endswith("configs/bilstm.yaml")
    assert {r["model"]: r["params"] for r in rows}["gru"] < {r["model"]: r["params"] for r in rows}["bilstm"]


def test_summarise_takes_the_median_over_repeats() -> None:
    rows = [_row("bilstm", True, r, ms) for r, ms in ((1, 1.0), (2, 3.0), (3, 2.0))]
    rows += [_row("bilstm", False, r, ms) for r, ms in ((1, 8.0), (2, 9.0), (3, 7.0))]
    s = summarise(rows)["bilstm"]
    assert (s["onednn_on_ms"], s["onednn_off_ms"], s["off_over_on"]) == (2.0, 8.0, 4.0)


def test_render_states_settings_ratios_and_profile_values() -> None:
    rows = [_row("bilstm", True, 1, 1.0), _row("bilstm", False, 1, 6.0),
            _row("gru", True, 1, 7.0), _row("gru", False, 1, 7.5)]
    md = render_latency_check_md(rows, {"bilstm": 1.25, "gru": 7.25, "mlp": 0.1}, HW)
    assert "| bilstm | 100 | 1.000 | 6.000 | 6.00 |" in md and "| gru | 100 | 7.000 | 7.500 | 1.07 |" in md
    assert "GRU / BiLSTM latency: 7.00 with oneDNN enabled, 1.25 with oneDNN disabled." in md
    assert "bilstm 1.250 ms, gru 7.250 ms." in md and "mlp" not in md
    assert "median over 1 repeats of the mean of 200 timed forward passes after 20 warm-up passes" in md
    assert "—" not in md


def test_write_latency_check_files_and_script(tmp_path: Path) -> None:
    configs, out = _configs(tmp_path), tmp_path / "out"
    spec = importlib.util.spec_from_file_location("gru_latency_check", REPO_ROOT / "scripts" / "gru_latency_check.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    paths = script.main(["--configs-dir", str(configs), "--results-root", str(tmp_path / "results"),
                         "--out-dir", str(out), "--repeats", "1"])
    assert [p.name for p in paths] == ["gru_latency_check.csv", "gru_latency_check.md"]
    with paths[0].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and {r["onednn"] for r in rows} == {"true", "false"}
    assert "For reference" not in paths[1].read_text(encoding="utf-8")       # no profile in this results root


def test_write_latency_check_defaults_to_the_analysis_folder(tmp_path: Path) -> None:
    paths = write_latency_check(_configs(tmp_path), results_root=tmp_path / "results", repeats=1)
    assert all(p.parent == tmp_path / "results" / "analysis" for p in paths)
