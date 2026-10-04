"""Model cost instrumentation: parameter count, size, inference latency, hardware (BUILD_SPEC 5.5, M2.2).

Single-thread, batch-size-1 CPU latency is the on-device proxy used in the comparison.
Latency does not depend on the weights, so untrained models can be profiled too.
"""

from __future__ import annotations

import copy
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameter elements (only those with ``requires_grad`` if ``trainable_only``)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def model_size_mb(model: nn.Module) -> float:
    """Size of all parameters and buffers stored as float32, in MiB (2**20 bytes)."""
    n_elements = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    return n_elements * 4 / 2**20


def best_available_device() -> str:
    """``"cuda"`` if available, else ``"mps"``, else ``"cpu"``."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so the clock measures the forward pass, not the kernel launch."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str | torch.device = "cpu",
    batch_size: int = 1,
    warmup: int = 20,
    iters: int = 200,
    threads: int | None = 1,
) -> dict:
    """Time forward passes of a copy of ``model`` on random input ``(batch_size, *input_shape)``.

    The copy runs in ``eval()`` mode under ``torch.inference_mode()``: ``warmup`` untimed
    passes, then ``iters`` passes timed one by one with ``time.perf_counter()``, synchronising
    CUDA/MPS before each clock read. On CPU, ``threads`` (if not None) sets
    ``torch.set_num_threads`` for the measurement and the previous value is restored
    afterwards, even on error. ``model`` itself is not moved or modified.

    Returns ``{'mean_ms', 'p50_ms', 'p95_ms', 'batch_size', 'device', 'threads', 'warmup',
    'iters'}``; ``threads`` is the CPU thread count used, or None on a GPU.
    """
    if batch_size < 1 or iters < 1 or warmup < 0:
        raise ValueError(f"need batch_size >= 1, iters >= 1, warmup >= 0; got {batch_size}, {iters}, {warmup}")
    device = torch.device(device)
    on_cpu = device.type == "cpu"
    previous_threads = torch.get_num_threads()
    if on_cpu and threads is not None:
        torch.set_num_threads(threads)
    try:
        threads_used = torch.get_num_threads() if on_cpu else None
        net = copy.deepcopy(model).to(device).eval()
        x = torch.randn((batch_size, *input_shape), device=device)
        times = []
        with torch.inference_mode():
            for _ in range(warmup):
                net(x)
            _synchronize(device)
            for _ in range(iters):
                start = time.perf_counter()
                net(x)
                _synchronize(device)
                times.append(time.perf_counter() - start)
    finally:
        if on_cpu and threads is not None:
            torch.set_num_threads(previous_threads)

    ms = np.asarray(times) * 1000.0
    return {
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "batch_size": int(batch_size),
        "device": str(device),
        "threads": threads_used,
        "warmup": int(warmup),
        "iters": int(iters),
    }


def _cpu_name() -> str:
    """Marketing name of the CPU (e.g. from the Windows registry), else ``platform.processor()``."""
    try:
        if sys.platform == "win32":
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, check=True, timeout=5)
            return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unknown"


def hardware_info(device: str | torch.device | None = None) -> dict:
    """Describe the machine: Python, torch, OS, CPU, the device used (default: best available), GPU name."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        gpu = "Apple GPU (MPS)"
    else:
        gpu = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device": str(device) if device is not None else best_available_device(),
        "gpu": gpu,
    }
