"""
hybrid_profiler.py — Fallback profiler using CUDA Events + Occupancy API + Analytical metrics.
Used when NCU is unavailable (permission denied, missing importer, etc.).

Combines 4 sources to approximate all 24 KernelMetrics fields:
  1. CUDA Event timing  → duration_us, speedup
  2. Analytical math     → mem_throughput_pct, compute_throughput_pct, dram_bytes, bandwidth
  3. CUDA Occupancy API  → sm_occupancy, achieved_occupancy (via compiled query program)
  4. Source-level analysis → stall estimates, instruction estimates, cycle counts
"""

from __future__ import annotations
import logging
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .metrics import KernelMetrics

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


class HybridProfiler:
    """
    Computes kernel metrics without NCU hardware counters.
    Drop-in replacement for NCURunner.profile() when permissions are blocked.
    """

    def __init__(self, config: dict, hw_spec: dict):
        self.config = config
        self.hw_spec = hw_spec

        # Hardware peaks from spec
        mem = hw_spec.get("memory", {})
        comp = hw_spec.get("compute", {})
        sm = hw_spec.get("sm", {})

        self.peak_mem_bw_bytes_per_sec = mem.get("hbm_bandwidth_tbs", 8.0) * 1e12
        self.peak_fp32_flops = comp.get("fp32_tflops", 67.0) * 1e12
        self.peak_fp16_flops = comp.get("fp16_tflops", 134.0) * 1e12
        self.sm_count = sm.get("count", 142)
        self.max_warps_per_sm = sm.get("max_warps_per_sm", 64)
        self.max_blocks_per_sm = sm.get("max_blocks_per_sm", 32)
        self.warp_size = sm.get("warp_size", 32)
        self.max_threads_per_sm = sm.get("max_threads_per_sm", 2048)
        self.shared_mem_per_sm = mem.get("shared_memory_per_sm_kb", 228) * 1024
        self.l2_cache_bytes = mem.get("l2_cache_mb", 96) * 1024 * 1024

        # Compilation settings (for occupancy query program)
        self.nvcc = "nvcc"
        self.cuda_arch = "sm_100a"
        self.nvcc_flags = [
            "-O3", f"-arch={self.cuda_arch}", "--use_fast_math", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
        ]

        # Cache device properties (queried once)
        self._device_props_cache: Optional[dict] = None

    # ── Main Entry Point ───────────────────────────────────────────────────

    def profile(
        self,
        kernel_src: str,
        timing_us: float,
        kernel_type: str,
        problem_shape: tuple,
        baseline_us: float = 0.0,
    ) -> Optional[KernelMetrics]:
        """
        Compute hybrid metrics from timing + analytical + source analysis.
        Returns a fully-populated KernelMetrics.
        """
        if timing_us <= 0:
            return None

        timing_sec = timing_us / 1e6

        # ── 1. Analytical: data movement ──────────────────────────────────
        read_bytes, write_bytes = self._compute_data_movement(kernel_type, problem_shape)
        total_bytes = read_bytes + write_bytes

        # ── 2. Analytical: FLOPs ──────────────────────────────────────────
        total_flops = self._estimate_flops(kernel_type, problem_shape)

        # ── 3. Bandwidth & throughput percentages ─────────────────────────
        achieved_bw = total_bytes / timing_sec
        mem_throughput_pct = min(100.0, (achieved_bw / self.peak_mem_bw_bytes_per_sec) * 100)

        achieved_flops = total_flops / timing_sec
        compute_throughput_pct = min(100.0, (achieved_flops / self.peak_fp32_flops) * 100)
        fp32_throughput_pct = compute_throughput_pct

        dram_read_bw_gbps = read_bytes / timing_sec / 1e9

        # ── 4. Occupancy (CUDA API or analytical) ─────────────────────────
        block_size, shared_mem = self._parse_launch_config(kernel_src)
        gpu_occupancy = self._query_occupancy_from_binary(kernel_src, block_size, shared_mem)
        if gpu_occupancy is None:
            gpu_occupancy = self._compute_theoretical_occupancy(block_size, shared_mem)

        # ── 5. Stall estimates ────────────────────────────────────────────
        stall_memory, stall_barrier, stall_no_inst, stall_mio = self._estimate_stalls(
            kernel_src, mem_throughput_pct, compute_throughput_pct, gpu_occupancy
        )

        # ── 6. Instruction estimates ──────────────────────────────────────
        inst_load, inst_store, inst_fp32, inst_fp16 = self._estimate_instructions(
            kernel_type, problem_shape
        )
        inst_executed = inst_load + inst_store + inst_fp32 + inst_fp16

        # ── 7. Cycle estimates ────────────────────────────────────────────
        sm_clock_ghz = 2.1  # approximate B200 boost clock
        elapsed_cycles = timing_us * 1e-6 * sm_clock_ghz * 1e9 * self.sm_count
        active_cycles = elapsed_cycles * gpu_occupancy / 100.0

        # ── 8. L2 cache estimates ─────────────────────────────────────────
        l2_hit_rate = self._estimate_l2_hit_rate(total_bytes, timing_sec)
        l2_sector_size = 32  # bytes per L2 sector
        l2_read_sectors = read_bytes / l2_sector_size
        l2_write_sectors = write_bytes / l2_sector_size

        # L1/L2 throughput: proportional to memory throughput
        l1_throughput_pct = min(100.0, mem_throughput_pct * 1.2)
        l2_throughput_pct = min(100.0, mem_throughput_pct * 1.1)

        # ── 9. Speedup ───────────────────────────────────────────────────
        speedup = baseline_us / timing_us if timing_us > 0 and baseline_us > 0 else 1.0

        metrics = KernelMetrics(
            mem_throughput_pct=round(mem_throughput_pct, 2),
            l1_throughput_pct=round(l1_throughput_pct, 2),
            l2_throughput_pct=round(l2_throughput_pct, 2),
            compute_throughput_pct=round(compute_throughput_pct, 2),
            fp32_throughput_pct=round(fp32_throughput_pct, 2),
            sm_occupancy=round(gpu_occupancy, 2),
            achieved_occupancy=round(gpu_occupancy * self.max_warps_per_sm / 100.0, 2),
            stall_memory=round(stall_memory, 2),
            stall_barrier=round(stall_barrier, 2),
            stall_no_instruction=round(stall_no_inst, 2),
            stall_mio_throttle=round(stall_mio, 2),
            l2_hit_rate=round(l2_hit_rate, 2),
            l2_read_sectors=round(l2_read_sectors, 2),
            l2_write_sectors=round(l2_write_sectors, 2),
            dram_read_bytes=round(read_bytes, 2),
            dram_write_bytes=round(write_bytes, 2),
            dram_read_bw_gbps=round(dram_read_bw_gbps, 2),
            inst_executed=round(inst_executed, 2),
            inst_fp32=round(inst_fp32, 2),
            inst_fp16=round(inst_fp16, 2),
            inst_load=round(inst_load, 2),
            inst_store=round(inst_store, 2),
            elapsed_cycles=round(elapsed_cycles, 2),
            active_cycles=round(active_cycles, 2),
            duration_us=timing_us,
            speedup=speedup,
        )

        logger.info(
            "Hybrid profiler: mem=%.1f%% compute=%.1f%% occ=%.1f%% "
            "stall_mem=%.1f%% bw=%.1f GB/s",
            mem_throughput_pct, compute_throughput_pct, gpu_occupancy,
            stall_memory, dram_read_bw_gbps,
        )
        return metrics

    # ── Data Movement Analysis ─────────────────────────────────────────────

    def _compute_data_movement(self, kernel_type: str, shape: tuple) -> tuple[float, float]:
        """Compute (read_bytes, write_bytes) from kernel type and problem shape."""

        if kernel_type == "add_rmsnorm":
            rows, hidden = shape[0], shape[1]
            n = rows * hidden
            nb = n // 16
            # Reads: input(bf16) + residual(bf16) + weight(bf16 per hidden)
            read_bytes = n * 2 + n * 2 + hidden * 2
            # Writes: output(bf16) + quantized(packed fp4 = N/2 bytes) + scales(fp8 = nb bytes)
            write_bytes = n * 2 + n // 2 + nb

        elif kernel_type == "silu_mul":
            n = self._shape_to_n(shape)
            nb = n // 16
            # Reads: gate(bf16) + up(bf16)
            read_bytes = n * 2 + n * 2
            # Writes: quantized(N/2) + scales(nb)
            write_bytes = n // 2 + nb

        elif kernel_type == "nvfp4_quantize":
            n = self._shape_to_n(shape)
            nb = n // 16
            # Reads: input(bf16)
            read_bytes = n * 2
            # Writes: packed(N/2) + scales(nb)
            write_bytes = n // 2 + nb

        else:
            n = self._shape_to_n(shape)
            read_bytes = n * 4
            write_bytes = n * 2

        return float(read_bytes), float(write_bytes)

    def _estimate_flops(self, kernel_type: str, shape: tuple) -> float:
        """Estimate total floating-point operations."""

        if kernel_type == "add_rmsnorm":
            rows, hidden = shape[0], shape[1]
            n = rows * hidden
            # add(N) + square(N) + reduce_sum(N) + rsqrt(rows) + norm_mul(N) + weight_mul(N) + quant(N)
            return float(n * 6 + rows)

        elif kernel_type == "silu_mul":
            n = self._shape_to_n(shape)
            # silu: exp(N) + add(N) + div(N) + mul(N); gate_mul(N); quant(N)
            return float(n * 6)

        elif kernel_type == "nvfp4_quantize":
            n = self._shape_to_n(shape)
            # absmax(N) + scale_div(N) + clamp(N) + pack(N)
            return float(n * 4)

        else:
            n = self._shape_to_n(shape)
            return float(n * 2)

    def _shape_to_n(self, shape: tuple) -> int:
        """Flatten shape tuple to total element count."""
        n = 1
        for s in shape:
            n *= s
        return n

    # ── Launch Config Parsing ──────────────────────────────────────────────

    def _parse_launch_config(self, kernel_src: str) -> tuple[int, int]:
        """Extract block size and shared memory from kernel source."""

        # Pattern 1: <<<grid, block>>> or <<<grid, block, smem>>>
        launch = re.search(
            r'<<<\s*[^,]+,\s*(\d+)\s*(?:,\s*(\d+))?\s*(?:,\s*\w+)?\s*>>>', kernel_src
        )
        if launch:
            block_size = int(launch.group(1))
            shared_mem = int(launch.group(2)) if launch.group(2) else 0
            return block_size, shared_mem

        # Pattern 2: dim3 block(x) or dim3 block(x,y) or dim3 block(x,y,z)
        dim3 = re.search(r'dim3\s+\w+\s*\(\s*(\d+)\s*(?:,\s*(\d+))?\s*(?:,\s*(\d+))?\s*\)', kernel_src)
        if dim3:
            x = int(dim3.group(1))
            y = int(dim3.group(2)) if dim3.group(2) else 1
            z = int(dim3.group(3)) if dim3.group(3) else 1
            return x * y * z, 0

        # Pattern 3: #define or constexpr for block size
        block_def = re.search(
            r'(?:#define\s+(?:BLOCK_SIZE|THREADS_PER_BLOCK|NUM_THREADS|BLOCK_DIM)\s+(\d+))|'
            r'(?:constexpr\s+int\s+(?:BLOCK_SIZE|THREADS_PER_BLOCK|NUM_THREADS|kBlockSize)\s*=\s*(\d+))',
            kernel_src
        )
        if block_def:
            block_size = int(block_def.group(1) or block_def.group(2))
        else:
            block_size = 256  # common default

        # Shared memory from __shared__ declarations
        shared_mem = self._estimate_shared_memory(kernel_src)

        return block_size, shared_mem

    def _estimate_shared_memory(self, kernel_src: str) -> int:
        """Estimate shared memory usage from __shared__ declarations."""
        total = 0
        # Match: __shared__ type name[size]
        for m in re.finditer(r'__shared__\s+(\w+)\s+\w+\[([^\]]+)\]', kernel_src):
            dtype = m.group(1)
            size_expr = m.group(2).strip()

            # Type size mapping
            type_sizes = {
                "float": 4, "double": 8, "int": 4, "unsigned": 4,
                "half": 2, "__half": 2, "__nv_bfloat16": 2,
                "float4": 16, "float2": 8, "int4": 16,
                "char": 1, "uint8_t": 1, "int8_t": 1,
                "uint32_t": 4, "int32_t": 4, "uint16_t": 2,
            }
            elem_size = type_sizes.get(dtype, 4)

            try:
                size = int(size_expr)
                total += size * elem_size
            except ValueError:
                total += 1024 * elem_size  # assume ~1K elements if can't parse

        # Also check extern __shared__
        if "extern __shared__" in kernel_src:
            total = max(total, 4096)  # conservative estimate for dynamic shared memory

        return total

    # ── Occupancy ──────────────────────────────────────────────────────────

    def _query_occupancy_from_binary(
        self, kernel_src: str, block_size: int, shared_mem: int
    ) -> Optional[float]:
        """
        Compile a small CUDA program that uses cudaOccupancyMaxActiveBlocksPerMultiprocessor
        to query actual occupancy (accounts for register pressure from compilation).
        Returns occupancy percentage or None if compilation/query fails.
        """
        # Find the __global__ function name
        global_funcs = re.findall(r'__global__\s+void\s+(\w+)\s*\(', kernel_src)
        if not global_funcs:
            return None

        kernel_name = global_funcs[0]

        # Build a small program that compiles the kernel and queries occupancy
        query_src = f"""
{kernel_src}

#include <cstdio>
#include <cuda_runtime.h>

int main() {{
    int num_blocks = 0;
    int block_size = {block_size};
    size_t shared_mem = {shared_mem};

    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, {kernel_name}, block_size, shared_mem);

    if (err != cudaSuccess) {{
        printf("occ_error\\n");
        return 1;
    }}

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int warps_per_block = (block_size + prop.warpSize - 1) / prop.warpSize;
    int active_warps = num_blocks * warps_per_block;
    int max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    float occupancy = (float)active_warps / (float)max_warps * 100.0f;

    printf("occ_pct: %.2f\\n", occupancy);
    printf("active_blocks: %d\\n", num_blocks);
    printf("sm_count: %d\\n", prop.multiProcessorCount);
    return 0;
}}
"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
                f.write(query_src)
                src_path = f.name

            bin_path = src_path.replace(".cu", "_occ")
            compile_cmd = [self.nvcc] + self.nvcc_flags + [src_path, "-o", bin_path]
            comp = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)

            if comp.returncode != 0:
                logger.debug("Occupancy query compilation failed: %s", comp.stderr[:200])
                return None

            run = subprocess.run([bin_path], capture_output=True, text=True, timeout=10)
            if run.returncode != 0:
                return None

            match = re.search(r"occ_pct:\s*([\d.]+)", run.stdout)
            if match:
                occ = float(match.group(1))
                logger.info("CUDA Occupancy API: %.1f%%", occ)
                return occ

        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("Occupancy query failed: %s", e)

        return None

    def _compute_theoretical_occupancy(self, block_size: int, shared_mem: int) -> float:
        """Compute theoretical occupancy from launch config + device specs (no GPU needed)."""

        warps_per_block = math.ceil(block_size / self.warp_size)

        # Limit 1: warp count
        max_blocks_by_warps = self.max_warps_per_sm // warps_per_block if warps_per_block > 0 else 0

        # Limit 2: shared memory
        if shared_mem > 0:
            max_blocks_by_smem = int(self.shared_mem_per_sm / shared_mem)
        else:
            max_blocks_by_smem = self.max_blocks_per_sm

        # Limit 3: hardware block limit
        max_blocks_hw = self.max_blocks_per_sm

        active_blocks = min(max_blocks_by_warps, max_blocks_by_smem, max_blocks_hw)
        active_warps = active_blocks * warps_per_block

        occupancy = (active_warps / self.max_warps_per_sm * 100.0) if self.max_warps_per_sm > 0 else 0
        return min(100.0, max(0.0, occupancy))

    # ── Stall Estimation ───────────────────────────────────────────────────

    def _estimate_stalls(
        self,
        kernel_src: str,
        mem_pct: float,
        compute_pct: float,
        occupancy_pct: float,
    ) -> tuple[float, float, float, float]:
        """
        Estimate stall percentages from source analysis + throughput ratio.

        Logic:
        - Memory-bound kernels → high stall_memory (waiting for DRAM)
        - Many __syncthreads → high stall_barrier
        - Low occupancy → high stall_no_instruction (not enough warps to hide latency)
        - High shared memory traffic → stall_mio_throttle
        """

        sync_count = kernel_src.count("__syncthreads")
        syncwarp_count = kernel_src.count("__syncwarp")
        barrier_points = sync_count + syncwarp_count
        shared_decls = kernel_src.count("__shared__")

        # Memory stall: dominant when memory-bound with underutilized compute
        if mem_pct > compute_pct and mem_pct > 30:
            stall_memory = min(80.0, (mem_pct - compute_pct) * 0.8 + 10.0)
        elif mem_pct > 50:
            stall_memory = min(60.0, mem_pct * 0.5)
        else:
            stall_memory = max(5.0, 25.0 - compute_pct * 0.2)

        # Barrier stall: from synchronization density
        stall_barrier = min(50.0, barrier_points * 3.5)
        if barrier_points == 0:
            stall_barrier = 1.0  # minimal baseline

        # No-instruction stall: low occupancy means fewer warps to schedule
        if occupancy_pct < 50:
            stall_no_inst = min(40.0, (100.0 - occupancy_pct) * 0.4)
        else:
            stall_no_inst = max(2.0, (100.0 - occupancy_pct) * 0.15)

        # MIO throttle: shared memory / L1 contention
        stall_mio = min(25.0, shared_decls * 2.5 + max(0, mem_pct - 60) * 0.15)

        return stall_memory, stall_barrier, stall_no_inst, stall_mio

    # ── Instruction Estimation ─────────────────────────────────────────────

    def _estimate_instructions(
        self, kernel_type: str, shape: tuple
    ) -> tuple[float, float, float, float]:
        """Estimate (inst_load, inst_store, inst_fp32, inst_fp16) from kernel type."""

        if kernel_type == "add_rmsnorm":
            rows, hidden = shape[0], shape[1]
            n = rows * hidden
            inst_load = float(n * 2 + hidden)     # input + residual + weight
            inst_store = float(n + n // 2 + n // 16)  # output + packed + scales
            inst_fp32 = float(n * 4)               # add, square, mul, norm
            inst_fp16 = float(n * 2)               # bf16 conversions

        elif kernel_type == "silu_mul":
            n = self._shape_to_n(shape)
            inst_load = float(n * 2)
            inst_store = float(n // 2 + n // 16)
            inst_fp32 = float(n * 4)               # exp, div, mul, gate_mul
            inst_fp16 = float(n)

        elif kernel_type == "nvfp4_quantize":
            n = self._shape_to_n(shape)
            inst_load = float(n)
            inst_store = float(n // 2 + n // 16)
            inst_fp32 = float(n * 3)               # absmax, scale, clamp
            inst_fp16 = float(n)

        else:
            n = self._shape_to_n(shape)
            inst_load = float(n)
            inst_store = float(n)
            inst_fp32 = float(n * 2)
            inst_fp16 = 0.0

        return inst_load, inst_store, inst_fp32, inst_fp16

    # ── L2 Cache Estimation ────────────────────────────────────────────────

    def _estimate_l2_hit_rate(self, total_bytes: float, timing_sec: float) -> float:
        """
        Estimate L2 hit rate from achieved vs theoretical bandwidth.
        If achieved BW exceeds peak DRAM BW → data must be hitting L2.
        If working set fits in L2 → high hit rate.
        """
        if timing_sec <= 0:
            return 0.0

        achieved_bw = total_bytes / timing_sec

        # If working set fits in L2 cache, expect high hit rate
        if total_bytes < self.l2_cache_bytes:
            return min(90.0, 50.0 + (1.0 - total_bytes / self.l2_cache_bytes) * 40.0)

        # If achieved BW > peak DRAM → L2 is absorbing some accesses
        if achieved_bw > self.peak_mem_bw_bytes_per_sec:
            ratio = achieved_bw / self.peak_mem_bw_bytes_per_sec
            return min(95.0, (1.0 - 1.0 / ratio) * 100)

        # Otherwise estimate from bandwidth utilization
        bw_ratio = achieved_bw / self.peak_mem_bw_bytes_per_sec
        return max(5.0, (1.0 - bw_ratio) * 25.0)
