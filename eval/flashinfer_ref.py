"""
flashinfer_ref.py — FlashInfer-based reference outputs and baseline timing.

Uses FlashInfer's production CUDA kernels on B200 as the ground-truth
reference for WaferBench NVFP4, matching the official evaluation methodology.
"""

from __future__ import annotations
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import flashinfer
    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False
    logger.warning("torch/flashinfer not available — will fall back to reference kernels")

# Timing constants matching ThunderKittens 2.0 / WaferBench convention
_WARMUP_ITERS = 500
_BENCH_ITERS  = 100


def available() -> bool:
    return _HAS_FLASHINFER and torch.cuda.is_available()


def measure_baseline(kernel_type: str, shape: tuple) -> Optional[float]:
    """Measure FlashInfer baseline timing in microseconds.

    Returns None if FlashInfer is not available.
    """
    if not available():
        return None

    try:
        if kernel_type == "add_rmsnorm":
            return _baseline_add_rmsnorm(shape)
        elif kernel_type == "silu_mul":
            return _baseline_silu_mul(shape)
        elif kernel_type == "nvfp4_quantize":
            return _baseline_nvfp4_quantize(shape)
        else:
            logger.warning("Unknown kernel_type for FlashInfer baseline: %s", kernel_type)
            return None
    except Exception as e:
        logger.warning("FlashInfer baseline measurement failed: %s", e)
        return None


def generate_reference(kernel_type: str, shape: tuple, seed: int = 42) -> Optional[dict]:
    """Generate reference output tensors using FlashInfer.

    Returns dict with input/output torch tensors, or None if unavailable.
    """
    if not available():
        return None

    try:
        if kernel_type == "add_rmsnorm":
            return _reference_add_rmsnorm(shape, seed)
        elif kernel_type == "silu_mul":
            return _reference_silu_mul(shape, seed)
        elif kernel_type == "nvfp4_quantize":
            return _reference_nvfp4_quantize(shape, seed)
        else:
            return None
    except Exception as e:
        logger.warning("FlashInfer reference generation failed: %s", e)
        return None


# ── Add + RMSNorm + FP4 Quant ────────────────────────────────────────────────

def _baseline_add_rmsnorm(shape: tuple) -> float:
    rows, hidden = shape
    torch.manual_seed(0)
    inp = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    res = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    w   = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")

    def run():
        # Must include FP4 quantization — our kernel does add+rmsnorm+fp4quant fused
        flashinfer.fused_add_rmsnorm(inp, res, w, eps=1e-6)
        flashinfer.fp4_quantize(inp)

    return _time_fn(run)


def _reference_add_rmsnorm(shape: tuple, seed: int) -> dict:
    rows, hidden = shape
    torch.manual_seed(seed)
    inp = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    res = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    w   = torch.ones(hidden, dtype=torch.bfloat16, device="cuda") + \
          torch.rand(hidden, dtype=torch.bfloat16, device="cuda") * 0.1

    # FlashInfer's fused_add_rmsnorm modifies inp in-place (residual_out = inp + res)
    inp_copy = inp.clone()
    res_copy = res.clone()
    flashinfer.fused_add_rmsnorm(inp_copy, res_copy, w, eps=1e-6)

    return {
        "input": inp, "residual": res, "weight": w,
        "residual_out": res_copy, "norm_out": inp_copy,
    }


# ── SiLU × Mul + FP4 Quant ──────────────────────────────────────────────────

def _baseline_silu_mul(shape: tuple) -> float:
    b, m, k = shape
    n = b * m * k
    torch.manual_seed(0)
    gate = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    up   = torch.randn(n, dtype=torch.bfloat16, device="cuda")

    def run():
        # Must include FP4 quantization — our kernel does silu*mul+fp4quant fused
        out = torch.nn.functional.silu(gate) * up
        flashinfer.fp4_quantize(out.view(b * m, k))

    return _time_fn(run)


def _reference_silu_mul(shape: tuple, seed: int) -> dict:
    b, m, k = shape
    n = b * m * k
    torch.manual_seed(seed)
    gate = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    up   = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    out  = torch.nn.functional.silu(gate) * up
    return {"gate": gate, "up": up, "output": out}


# ── NVFP4 Quantize ──────────────────────────────────────────────────────────

def _baseline_nvfp4_quantize(shape: tuple) -> float:
    m, k = shape
    torch.manual_seed(0)
    inp = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

    def run():
        flashinfer.fp4_quantize(inp)

    return _time_fn(run)


def _reference_nvfp4_quantize(shape: tuple, seed: int) -> dict:
    m, k = shape
    torch.manual_seed(seed)
    inp = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    packed, scales = flashinfer.fp4_quantize(inp)
    return {"input": inp, "packed": packed, "scales": scales}


# ── Timing utility ───────────────────────────────────────────────────────────

def _time_fn(fn, warmup: int = _WARMUP_ITERS, iters: int = _BENCH_ITERS) -> float:
    """Time a function using CUDA events. Returns microseconds per call."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    return ms * 1000.0 / iters  # convert ms → µs per iteration
