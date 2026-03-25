"""
benchmark.py — Timing harness for WaferBench problem shapes.
"""

from __future__ import annotations
import logging
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


def geometric_mean(values: list) -> float:
    if not values:
        return 1.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


class Benchmarker:

    def __init__(self, config: dict, kernel_type: str = "add_rmsnorm"):
        self.warmup      = config["eval"]["benchmark_warmup"]
        self.iters       = config["eval"]["benchmark_iters"]
        self.kernel_type = kernel_type
        self.nvcc_flags  = [
            "-O3", "-arch=sm_100a", "--use_fast_math", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
        ]

    # ── Harness generators (one per kernel type) ──────────────────────────────

    def _build_harness(self, shape: tuple) -> str:
        if self.kernel_type == "add_rmsnorm":
            return self._harness_add_rmsnorm(shape)
        elif self.kernel_type == "silu_mul":
            return self._harness_silu_mul(shape)
        elif self.kernel_type == "nvfp4_quantize":
            return self._harness_nvfp4_quantize(shape)
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

    def _timing_footer(self, launch_call: str, iters: int, warmup: int) -> str:
        """Shared dual-timing footer (event + wall clock) injected into each harness."""
        return f"""
    for(int i=0;i<{warmup};++i) {launch_call}
    cudaStreamSynchronize(s);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<{iters};++i) {launch_call}
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms_event=0; cudaEventElapsedTime(&ms_event,t0,t1);
    float us_event=ms_event*1000.f/{iters};
    cudaDeviceSynchronize();
    struct timespec ts0,ts1; clock_gettime(CLOCK_MONOTONIC,&ts0);
    for(int i=0;i<{iters};++i) {launch_call}
    cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC,&ts1);
    double wall_ns=(ts1.tv_sec-ts0.tv_sec)*1e9+(ts1.tv_nsec-ts0.tv_nsec);
    float us_wall=(float)(wall_ns/1000.0/{iters});
    float ratio=(us_event>0)?us_wall/us_event:1.f;
    printf("timing_us: %.3f\\n",us_event);
    printf("timing_wall_us: %.3f\\n",us_wall);
    printf("timing_ratio: %.3f\\n",ratio);
    if(ratio>1.5f) printf("TIMING_ANOMALY: event=%.3fus wall=%.3fus ratio=%.2fx\\n",us_event,us_wall,ratio);
"""

    def _harness_add_rmsnorm(self, shape: tuple) -> str:
        rows, hidden = shape
        n, nb = rows * hidden, rows * hidden // 16
        launch = f"launch_fused_add_rmsnorm_nvfp4(di,dr,dw,dro,dq,ds,rows,hidden,s);"
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <time.h>
void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, unsigned char*, __nv_bfloat16*, int, int, cudaStream_t);
int main() {{
    const int rows={rows}, hidden={hidden}, N={n}, nb={nb};
    __nv_bfloat16 *di,*dr,*dw,*dro,*ds; unsigned char *dq;
    cudaMalloc(&di,N*2); cudaMalloc(&dr,N*2); cudaMalloc(&dw,hidden*2);
    cudaMalloc(&dro,N*2); cudaMalloc(&dq,N/2); cudaMalloc(&ds,nb*2);
    cudaStream_t s; cudaStreamCreate(&s);
    {self._timing_footer(launch, self.iters, self.warmup)}
    return 0;
}}
"""

    def _harness_silu_mul(self, shape: tuple) -> str:
        b, m, k = shape
        n = b * m * k
        launch = f"launch_silu_mul_bf16(dg,du,dout,N,s);"
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <time.h>
void launch_silu_mul_bf16(
    const __nv_bfloat162*, const __nv_bfloat162*, __nv_bfloat162*, int, cudaStream_t);
int main() {{
    const int N={n};
    __nv_bfloat162 *dg,*du,*dout;
    cudaMalloc(&dg,N*2); cudaMalloc(&du,N*2); cudaMalloc(&dout,N*2);
    cudaStream_t s; cudaStreamCreate(&s);
    {self._timing_footer(launch, self.iters, self.warmup)}
    return 0;
}}
"""

    def _harness_nvfp4_quantize(self, shape: tuple) -> str:
        m, k = shape
        n = m * k
        nb = n // 16
        launch = f"launch_nvfp4_quantize_bf16(din,dpk,dsc,N,s);"
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
void launch_nvfp4_quantize_bf16(
    const __nv_bfloat16*, uint8_t*, __nv_bfloat16*, int, cudaStream_t);
int main() {{
    const int N={n}, nb={nb};
    __nv_bfloat16 *din,*dsc; uint8_t *dpk;
    cudaMalloc(&din,N*2); cudaMalloc(&dpk,N/2); cudaMalloc(&dsc,nb*2);
    cudaStream_t s; cudaStreamCreate(&s);
    {self._timing_footer(launch, self.iters, self.warmup)}
    return 0;
}}
"""

    # ── Compile + time ────────────────────────────────────────────────────────

    def _compile_and_time(self, kernel_src: str, shape: tuple) -> Optional[float]:
        harness  = self._build_harness(shape)
        combined = kernel_src + "\n\n" + harness
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "bench.cu"
            exe = Path(tmpdir) / "bench"
            src.write_text(combined)
            cmd = ["nvcc"] + self.nvcc_flags + [str(src), "-o", str(exe)]
            r   = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return None
            r2     = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            output = r2.stdout
            if "TIMING_ANOMALY" in output:
                logger.warning("Timing anomaly detected (event/wall ratio > 1.5x): %s",
                               re.search(r"TIMING_ANOMALY:.*", output).group(0))
            m = re.search(r"timing_us:\s*([\d.]+)", output)
            return float(m.group(1)) if m else None

    def benchmark(self, kernel_src: str, baseline_us_per_shape: dict) -> dict:
        results  = {}
        speedups = []
        for shape, baseline in baseline_us_per_shape.items():
            t_us = self._compile_and_time(kernel_src, shape)
            if t_us is not None:
                speedup = baseline / t_us
                speedups.append(speedup)
                results[shape] = {"timing_us": t_us, "speedup": speedup, "baseline_us": baseline}
                logger.info("Shape %s: %.2f us → %.3fx", shape, t_us, speedup)
            else:
                results[shape] = {"timing_us": None, "speedup": 0.0, "baseline_us": baseline}
                logger.warning("Shape %s: benchmark failed", shape)
        results["geomean_speedup"] = geometric_mean(speedups)
        results["speedups"]        = speedups
        logger.info("Geometric mean speedup: %.3fx", results["geomean_speedup"])
        return results
