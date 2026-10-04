"""Tests for har.eval.profiling."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from har.eval.profiling import count_params, hardware_info, measure_latency, model_size_mb

LATENCY_KEYS = {"mean_ms", "p50_ms", "p95_ms", "batch_size", "device", "threads", "warmup", "iters"}
FAST = {"warmup": 2, "iters": 5}


class ThreadProbe(nn.Module):
    """Records the CPU thread count seen during forward."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.seen: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(torch.get_num_threads())
        return self.linear(x)


class Exploding(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")


def test_count_params_linear_10_to_5_has_55() -> None:
    model = nn.Linear(10, 5)
    assert count_params(model) == 55
    model.weight.requires_grad_(False)
    assert count_params(model) == 5                       # bias only
    assert count_params(model, trainable_only=False) == 55


def test_model_size_counts_parameters_and_buffers_as_float32() -> None:
    assert model_size_mb(nn.Linear(10, 5)) == pytest.approx(55 * 4 / 2**20)
    # BatchNorm1d(4): weight + bias (8) and running_mean, running_var, num_batches_tracked (9).
    assert model_size_mb(nn.BatchNorm1d(4)) == pytest.approx(17 * 4 / 2**20)


def test_latency_dict_has_positive_timings() -> None:
    out = measure_latency(nn.Linear(16, 6), (16,), batch_size=3, **FAST)
    assert set(out) == LATENCY_KEYS
    assert out["mean_ms"] > 0 and out["p50_ms"] > 0 and out["p95_ms"] >= out["p50_ms"]
    assert (out["batch_size"], out["device"], out["threads"], out["iters"]) == (3, "cpu", 1, 5)
    json.dumps(out)


def test_thread_count_is_set_during_and_restored_after() -> None:
    before = torch.get_num_threads()
    target = 2 if before == 1 else 1
    probe = ThreadProbe()
    out = measure_latency(probe, (4,), threads=target, **FAST)
    assert torch.get_num_threads() == before
    assert out["threads"] == target
    assert probe.seen == []                     # the original model was never called (a copy was)


def test_thread_count_restored_even_when_the_model_fails() -> None:
    before = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="boom"):
        measure_latency(Exploding(), (4,), threads=2 if before == 1 else 1, **FAST)
    assert torch.get_num_threads() == before


def test_original_model_is_not_modified() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5)).train()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    measure_latency(model, (4,), batch_size=2, **FAST)
    assert model.training
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())


def test_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        measure_latency(nn.Linear(4, 2), (4,), batch_size=0)
    with pytest.raises(ValueError):
        measure_latency(nn.Linear(4, 2), (4,), iters=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_latency() -> None:
    out = measure_latency(nn.Linear(16, 6), (16,), device="cuda", **FAST)
    assert out["device"].startswith("cuda") and out["threads"] is None and out["mean_ms"] > 0


def test_hardware_info_is_json_serialisable() -> None:
    info = hardware_info()
    assert {"python", "torch", "os", "cpu", "processor", "cpu_count", "device", "gpu"} <= set(info)
    assert info["device"] in {"cuda", "mps", "cpu"}
    assert hardware_info("cpu")["device"] == "cpu"
    json.dumps(info)
