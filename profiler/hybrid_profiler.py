"""
hybrid_profiler.py — Fallback profiler using CUDA Events + Occupancy API + Analytical metrics.
Used when NCU is unavailable (permission denied, missing importer, etc.).

Combines 4 sources to approximate all 24 KernelMetrics fields:
  1. CUDA Event timing   → duration_us, speedup
  2. Analytical math      → mem_throughput_pct, compute_throughput_pct, dram_bytes, bandwidth
  3. CUDA Occupancy API   → sm_occupancy, achieved_occupancy (via compiled query program)
  4. Source-level analysis → adjusts bytes/FLOPs based on actual kernel patterns,
                             differentiates stall profiles per implementation
"""

from __future__ import annotations
import logging
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .metrics import KernelMetrics, CompilerMetrics

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


# ── Source Pattern Analysis ────────────────────────────────────────────────

@dataclass
class SourceProfile:
    """Extracted source-level features that differentiate kernel implementations."""
    # Memory patterns
    vectorized_loads: int = 0       # float4/uint4/int4/bf16x8 loads
    vectorized_stores: int = 0      # float4/uint4/int4 stores
    scalar_loads: int = 0           # scalar global reads
    scalar_stores: int = 0          # scalar global writes
    ldg_hints: int = 0              # __ldg() read-only cache hints
    smem_declarations: int = 0      # __shared__ declarations
    smem_total_bytes: int = 0       # estimated shared memory usage
    smem_reads: int = 0             # reads from shared memory arrays
    smem_writes: int = 0            # writes to shared memory arrays
    cp_async: int = 0               # cp.async prefetch instructions
    tma_ops: int = 0                # TMA load/store operations

    # Compute patterns
    syncthreads: int = 0            # __syncthreads()
    syncwarp: int = 0               # __syncwarp()
    shfl_ops: int = 0               # __shfl_* warp shuffles
    pragma_unroll: int = 0          # #pragma unroll
    fast_math: int = 0              # __expf, __rsqrtf, __fmaf_rn, etc.
    fma_ops: int = 0                # fused multiply-add patterns
    reduction_ops: int = 0          # warp-level reductions

    # Structure
    num_global_funcs: int = 0       # __global__ functions
    num_device_funcs: int = 0       # __device__ functions
    total_lines: int = 0            # source lines (complexity proxy)
    loop_count: int = 0             # for/while loops
    branch_count: int = 0           # if/else branches

    @property
    def has_vectorized_access(self) -> bool:
        return self.vectorized_loads > 0 or self.vectorized_stores > 0

    @property
    def has_smem_caching(self) -> bool:
        return self.smem_declarations > 0 and self.smem_reads > 0

    @property
    def has_prefetch(self) -> bool:
        return self.cp_async > 0 or self.tma_ops > 0

    @property
    def has_warp_level_ops(self) -> bool:
        return self.shfl_ops > 0 or self.reduction_ops > 0

    @property
    def sync_density(self) -> float:
        """Synchronization points per 100 lines of code."""
        total_syncs = self.syncthreads + self.syncwarp
        return (total_syncs / max(self.total_lines, 1)) * 100


def analyze_source(src: str) -> SourceProfile:
    """Extract source-level features from kernel CUDA code."""
    p = SourceProfile()

    # Memory patterns
    p.vectorized_loads = len(re.findall(
        r'(?:float4|uint4|int4|ulonglong2|uint2)\s+\w+\s*=\s*\*?\s*(?:reinterpret_cast|__ldg|\()',
        src)) + src.count("load_bf16x8") + src.count("load_128") + len(re.findall(
        r'\*\s*reinterpret_cast\s*<\s*(?:float4|uint4|int4|ulonglong2)\s*\*\s*>', src))
    p.vectorized_stores = len(re.findall(
        r'\*\s*reinterpret_cast\s*<\s*(?:float4|uint4|int4|ulonglong2)\s*\*\s*>\s*\(', src
    )) + src.count("store_bf16x8") + src.count("store_128")
    p.scalar_loads = len(re.findall(r'\w+\s*\[\s*\w+\s*\]', src))  # rough count
    p.scalar_stores = len(re.findall(r'\w+\s*\[\s*\w+\s*\]\s*=', src))
    p.ldg_hints = src.count("__ldg")
    p.cp_async = src.count("cp.async") + src.count("cp_async")
    p.tma_ops = src.count("tma_load") + src.count("tma_store") + src.count("tcgen05")

    # Shared memory
    p.smem_declarations = src.count("__shared__")
    p.smem_total_bytes = _estimate_shared_memory(src)
    # Count reads/writes to known smem variable patterns
    smem_vars = re.findall(r'__shared__\s+\w+\s+(\w+)\s*\[', src)
    for var in smem_vars:
        p.smem_reads += len(re.findall(rf'\b{var}\s*\[', src)) - 1  # -1 for declaration
        p.smem_writes += len(re.findall(rf'\b{var}\s*\[.*\]\s*=', src))

    # Compute patterns
    p.syncthreads = src.count("__syncthreads")
    p.syncwarp = src.count("__syncwarp")
    p.shfl_ops = len(re.findall(r'__shfl_\w+', src))
    p.pragma_unroll = src.count("#pragma unroll")
    p.fast_math = len(re.findall(
        r'__(?:expf|logf|rsqrtf|sqrtf|fmaf_rn|fdividef|powf|sinf|cosf)\s*\(', src))
    p.fma_ops = len(re.findall(r'(?:__fmaf_rn|fmaf|__fmul_rn)\s*\(', src))
    p.reduction_ops = len(re.findall(
        r'(?:__reduce_\w+|warpReduceSum|warpReduce|blockReduce)\s*[(<]', src))

    # Structure
    p.num_global_funcs = len(re.findall(r'__global__\s+void\s+\w+', src))
    p.num_device_funcs = len(re.findall(r'__device__\s+(?:void|float|int|__nv_bfloat16)\s+\w+', src))
    p.total_lines = len(src.splitlines())
    p.loop_count = len(re.findall(r'\b(?:for|while)\s*\(', src))
    p.branch_count = len(re.findall(r'\bif\s*\(', src))

    return p


def _estimate_shared_memory(kernel_src: str) -> int:
    """Estimate shared memory usage from __shared__ declarations."""
    total = 0
    type_sizes = {
        "float": 4, "double": 8, "int": 4, "unsigned": 4,
        "half": 2, "__half": 2, "__nv_bfloat16": 2,
        "float4": 16, "float2": 8, "int4": 16,
        "char": 1, "uint8_t": 1, "int8_t": 1,
        "uint32_t": 4, "int32_t": 4, "uint16_t": 2,
    }
    for m in re.finditer(r'__shared__\s+(\w+)\s+\w+\[([^\]]+)\]', kernel_src):
        dtype, size_expr = m.group(1), m.group(2).strip()
        elem_size = type_sizes.get(dtype, 4)
        try:
            total += int(size_expr) * elem_size
        except ValueError:
            total += 1024 * elem_size
    if "extern __shared__" in kernel_src:
        total = max(total, 4096)
    return total


# ── Main Profiler Class ───────────────────────────────────────────────────

class HybridProfiler:
    """
    Computes kernel metrics without NCU hardware counters.
    Drop-in replacement for NCURunner.profile() when permissions are blocked.

    Uses source-level analysis to differentiate kernel implementations:
    different kernels for the same problem get DIFFERENT metrics based on
    their actual code patterns (vectorization, shared memory, syncs, etc.)
    """

    def __init__(self, config: dict, hw_spec: dict):
        self.config = config
        self.hw_spec = hw_spec

        mem = hw_spec.get("memory", {})
        comp = hw_spec.get("compute", {})
        sm = hw_spec.get("sm", {})

        # Use achievable peaks (75% of theoretical) for realistic percentages
        achievable_ratio = 0.75
        self.peak_mem_bw_bytes_per_sec = mem.get("hbm_bandwidth_tbs", 8.0) * 1e12 * achievable_ratio
        self.peak_fp32_flops = comp.get("fp32_tflops", 67.0) * 1e12 * achievable_ratio
        self.peak_fp16_flops = comp.get("fp16_tflops", 134.0) * 1e12 * achievable_ratio
        self.sm_count = sm.get("count", 148)
        self.max_warps_per_sm = sm.get("max_warps_per_sm", 64)
        self.max_blocks_per_sm = sm.get("max_blocks_per_sm", 32)
        self.warp_size = sm.get("warp_size", 32)
        self.max_threads_per_sm = sm.get("max_threads_per_sm", 2048)
        self.shared_mem_per_sm = mem.get("shared_memory_per_sm_kb", 228) * 1024
        self.l2_cache_bytes = mem.get("l2_cache_mb", 126) * 1024 * 1024

        self.nvcc = "nvcc"
        self.cuda_arch = "sm_100a"
        self.nvcc_flags = [
            "-O3", f"-arch={self.cuda_arch}", "--use_fast_math", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
        ]

    # ── Main Entry Point ───────────────────────────────────────────────────

    def profile(
        self,
        kernel_src: str,
        timing_us: float,
        kernel_type: str,
        problem_shape: tuple,
        baseline_us: float = 0.0,
        compiler_metrics: Optional[CompilerMetrics] = None,
    ) -> Optional[KernelMetrics]:
        if timing_us <= 0:
            return None

        timing_sec = timing_us / 1e6
        cm = compiler_metrics or CompilerMetrics()

        # ── 0. Analyze source patterns (fallback for missing compiler data) ─
        sp = analyze_source(kernel_src)

        # ── 1. Data movement (analytical baseline) ──────────────────────────
        base_read, base_write = self._compute_data_movement(kernel_type, problem_shape)
        total_bytes = base_read + base_write

        # ── 2. ACHIEVED bandwidth from TIMING (primary signal) ──────────────
        # This is the REAL metric — different candidates with different timings
        # get genuinely different bandwidth numbers
        achieved_bw = total_bytes / timing_sec
        mem_throughput_pct = min(100.0, (achieved_bw / self.peak_mem_bw_bytes_per_sec) * 100)
        dram_read_bw_gbps = base_read / timing_sec / 1e9

        # ── 3. Compute throughput from SASS or analytical ───────────────────
        if cm.sass_total_instructions > 0:
            # Real FLOPs from SASS disassembly
            total_flops = (cm.sass_ffma * 2 + cm.sass_fadd + cm.sass_fmul +
                           cm.sass_hfma2 * 4 + cm.sass_mufu)
            achieved_flops_rate = total_flops / timing_sec if total_flops > 0 else 0
        else:
            base_flops = self._estimate_flops(kernel_type, problem_shape)
            achieved_flops_rate = base_flops / timing_sec

        compute_throughput_pct = min(100.0, (achieved_flops_rate / self.peak_fp32_flops) * 100)
        fp32_throughput_pct = compute_throughput_pct

        # ── 4. Occupancy from CUDA API (uses real register count) ───────────
        block_size, shared_mem = self._parse_launch_config(kernel_src)
        # Override shared_mem with compiler's exact value if available
        if cm.static_smem_bytes > 0:
            shared_mem = cm.static_smem_bytes
        gpu_occupancy = self._query_occupancy_from_binary(kernel_src, block_size, shared_mem)
        if gpu_occupancy is None:
            gpu_occupancy = self._compute_theoretical_occupancy(
                block_size, shared_mem, cm.registers_per_thread
            )

        # ── 5. Stall estimates from SASS + timing ──────────────────────────
        stall_memory, stall_barrier, stall_no_inst, stall_mio = self._estimate_stalls_from_compiler(
            cm, sp, mem_throughput_pct, compute_throughput_pct, gpu_occupancy
        )

        # ── 6. Instructions from SASS (real) or analytical (fallback) ──────
        if cm.sass_total_instructions > 0:
            inst_load = float(cm.sass_ldg_32 + cm.sass_ldg_64 + cm.sass_ldg_128 + cm.sass_lds)
            inst_store = float(cm.sass_stg_32 + cm.sass_stg_64 + cm.sass_stg_128 + cm.sass_sts)
            inst_fp32 = float(cm.sass_ffma + cm.sass_fadd + cm.sass_fmul + cm.sass_mufu)
            inst_fp16 = float(cm.sass_hfma2)
            inst_executed = float(cm.sass_total_instructions)
        else:
            inst_load, inst_store, inst_fp32, inst_fp16 = self._estimate_instructions(
                kernel_type, problem_shape, sp
            )
            inst_executed = inst_load + inst_store + inst_fp32 + inst_fp16

        # ── 7. Cycle estimates ─────────────────────────────────────────────
        sm_clock_ghz = 2.1
        elapsed_cycles = timing_us * 1e-6 * sm_clock_ghz * 1e9 * self.sm_count
        active_cycles = elapsed_cycles * gpu_occupancy / 100.0

        # ── 8. L2 cache estimates ──────────────────────────────────────────
        l2_hit_rate = self._estimate_l2_hit_rate(sp, total_bytes, total_bytes, timing_sec)
        l2_sector_size = 32
        l2_read_sectors = base_read / l2_sector_size
        l2_write_sectors = base_write / l2_sector_size

        l1_throughput_pct = min(100.0, mem_throughput_pct * 1.2)
        l2_throughput_pct = min(100.0, mem_throughput_pct * 1.1)

        # ── 9. Speedup ────────────────────────────────────────────────────
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
            dram_read_bytes=round(base_read, 2),
            dram_write_bytes=round(base_write, 2),
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

        # Store compiler metrics on the KernelMetrics for the reflection prompt
        metrics._compiler_metrics = cm

        # Build log message with real data
        log_parts = [
            f"mem={mem_throughput_pct:.1f}%",
            f"compute={compute_throughput_pct:.1f}%",
            f"occ={gpu_occupancy:.1f}%",
            f"stall_mem={stall_memory:.1f}%",
            f"stall_bar={stall_barrier:.1f}%",
            f"bw={dram_read_bw_gbps:.1f} GB/s",
        ]
        if cm.registers_per_thread > 0:
            log_parts.append(f"regs={cm.registers_per_thread}")
        if cm.has_spills:
            log_parts.append(f"SPILLS={cm.spill_stores_bytes}B")
        if cm.sass_total_instructions > 0:
            log_parts.append(f"sass={cm.sass_total_instructions}")
            log_parts.append(f"vec_ld%={cm.vectorized_load_pct:.0f}")

        logger.info("Hybrid profiler: %s", " ".join(log_parts))
        return metrics

    # ── Source-Aware Efficiency Factors ─────────────────────────────────────

    def _memory_efficiency_factor(self, sp: SourceProfile) -> float:
        """
        Compute how efficiently the kernel accesses memory.
        Returns a multiplier for effective DRAM bytes:
          < 1.0 = kernel is efficient (fewer actual DRAM bytes than logical)
          > 1.0 = kernel is inefficient (more DRAM bytes due to poor coalescing)
          = 1.0 = baseline (scalar, no optimization)

        Different implementations get DIFFERENT factors:
        - Vectorized loads → 0.7-0.85x (fewer transactions, better coalescing)
        - Shared memory caching → 0.5-0.7x (reuse from SMEM, fewer DRAM reads)
        - TMA/cp.async → 0.6-0.75x (hardware prefetch, overlap)
        - __ldg hints → 0.9x (read-only cache path)
        - No optimization → 1.0x-1.2x (possibly uncoalesced)
        """
        factor = 1.0

        # Vectorized loads reduce memory transactions
        if sp.vectorized_loads > 0:
            vec_ratio = sp.vectorized_loads / max(sp.scalar_loads, 1)
            factor *= max(0.65, 1.0 - vec_ratio * 0.25)

        # Vectorized stores
        if sp.vectorized_stores > 0:
            factor *= 0.9

        # Shared memory caching: data read from DRAM once, reused from SMEM
        if sp.has_smem_caching:
            # More smem reads relative to declarations → more reuse
            reuse_ratio = sp.smem_reads / max(sp.smem_declarations, 1)
            factor *= max(0.5, 1.0 - min(reuse_ratio * 0.05, 0.4))

        # TMA / cp.async prefetching
        if sp.has_prefetch:
            factor *= 0.75

        # __ldg read-only cache hints
        if sp.ldg_hints > 0:
            factor *= max(0.85, 1.0 - sp.ldg_hints * 0.01)

        return max(0.4, min(1.3, factor))

    def _compute_efficiency_factor(self, sp: SourceProfile) -> float:
        """
        Compute how the kernel's compute intensity differs from baseline.
        Returns a multiplier for effective FLOPs:
          > 1.0 = more compute work (warp shuffles, extra reductions)
          < 1.0 = fewer FLOPs (fast math, fused ops)
          = 1.0 = baseline
        """
        factor = 1.0

        # Warp shuffles add compute work (but replace shared memory reductions)
        if sp.shfl_ops > 0:
            factor += sp.shfl_ops * 0.02

        # Fast math intrinsics: fewer cycles per FLOP
        if sp.fast_math > 0:
            factor *= max(0.7, 1.0 - sp.fast_math * 0.03)

        # FMA: fused multiply-add = 2 FLOPs in 1 instruction
        if sp.fma_ops > 0:
            factor += sp.fma_ops * 0.01

        # Loop unrolling: more instructions, better ILP
        if sp.pragma_unroll > 0:
            factor *= 1.0 + sp.pragma_unroll * 0.05

        # Extra device functions suggest more complex computation
        if sp.num_device_funcs > 2:
            factor *= 1.0 + (sp.num_device_funcs - 2) * 0.05

        return max(0.5, min(2.0, factor))

    # ── Data Movement Analysis ─────────────────────────────────────────────

    def _compute_data_movement(self, kernel_type: str, shape: tuple) -> tuple[float, float]:
        """Compute BASE (read_bytes, write_bytes) from kernel type and problem shape."""
        if kernel_type == "add_rmsnorm":
            rows, hidden = shape[0], shape[1]
            n = rows * hidden
            nb = n // 16
            read_bytes = n * 2 + n * 2 + hidden * 2
            write_bytes = n * 2 + n // 2 + nb
        elif kernel_type == "silu_mul":
            n = self._shape_to_n(shape)
            nb = n // 16
            read_bytes = n * 2 + n * 2
            write_bytes = n // 2 + nb
        elif kernel_type == "nvfp4_quantize":
            n = self._shape_to_n(shape)
            nb = n // 16
            read_bytes = n * 2
            write_bytes = n // 2 + nb
        else:
            n = self._shape_to_n(shape)
            read_bytes = n * 4
            write_bytes = n * 2
        return float(read_bytes), float(write_bytes)

    def _estimate_flops(self, kernel_type: str, shape: tuple) -> float:
        """Estimate BASE floating-point operations."""
        if kernel_type == "add_rmsnorm":
            rows, hidden = shape[0], shape[1]
            n = rows * hidden
            return float(n * 6 + rows)
        elif kernel_type == "silu_mul":
            n = self._shape_to_n(shape)
            return float(n * 6)
        elif kernel_type == "nvfp4_quantize":
            n = self._shape_to_n(shape)
            return float(n * 4)
        else:
            n = self._shape_to_n(shape)
            return float(n * 2)

    def _shape_to_n(self, shape: tuple) -> int:
        n = 1
        for s in shape:
            n *= s
        return n

    # ── Launch Config Parsing ──────────────────────────────────────────────

    def _parse_launch_config(self, kernel_src: str) -> tuple[int, int]:
        """Extract block size and shared memory from kernel source."""
        launch = re.search(
            r'<<<\s*[^,]+,\s*(\d+)\s*(?:,\s*(\d+))?\s*(?:,\s*\w+)?\s*>>>', kernel_src
        )
        if launch:
            block_size = int(launch.group(1))
            shared_mem = int(launch.group(2)) if launch.group(2) else 0
            return block_size, shared_mem

        dim3 = re.search(r'dim3\s+\w+\s*\(\s*(\d+)\s*(?:,\s*(\d+))?\s*(?:,\s*(\d+))?\s*\)', kernel_src)
        if dim3:
            x = int(dim3.group(1))
            y = int(dim3.group(2)) if dim3.group(2) else 1
            z = int(dim3.group(3)) if dim3.group(3) else 1
            return x * y * z, 0

        block_def = re.search(
            r'(?:#define\s+(?:BLOCK_SIZE|THREADS_PER_BLOCK|NUM_THREADS|BLOCK_DIM)\s+(\d+))|'
            r'(?:constexpr\s+int\s+(?:BLOCK_SIZE|THREADS_PER_BLOCK|NUM_THREADS|kBlockSize)\s*=\s*(\d+))',
            kernel_src
        )
        block_size = int(block_def.group(1) or block_def.group(2)) if block_def else 256
        shared_mem = _estimate_shared_memory(kernel_src)
        return block_size, shared_mem

    # ── Occupancy ──────────────────────────────────────────────────────────

    def _query_occupancy_from_binary(
        self, kernel_src: str, block_size: int, shared_mem: int
    ) -> Optional[float]:
        """Query real occupancy via CUDA API (accounts for register pressure)."""
        global_funcs = re.findall(r'__global__\s+void\s+(\w+)\s*\(', kernel_src)
        if not global_funcs:
            return None

        kernel_name = global_funcs[0]
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
    return 0;
}}
"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
                f.write(query_src)
                src_path = f.name
            bin_path = src_path.replace(".cu", "_occ")
            comp = subprocess.run(
                [self.nvcc] + self.nvcc_flags + [src_path, "-o", bin_path],
                capture_output=True, text=True, timeout=60,
            )
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

    def _compute_theoretical_occupancy(
        self, block_size: int, shared_mem: int, registers_per_thread: int = 0
    ) -> float:
        warps_per_block = math.ceil(block_size / self.warp_size)
        max_by_warps = self.max_warps_per_sm // warps_per_block if warps_per_block > 0 else 0
        max_by_smem = int(self.shared_mem_per_sm / shared_mem) if shared_mem > 0 else self.max_blocks_per_sm

        # Register-based occupancy limit (B200: 65536 registers per SM)
        regs_per_sm = 65536
        if registers_per_thread > 0:
            regs_per_block = registers_per_thread * block_size
            max_by_regs = regs_per_sm // regs_per_block if regs_per_block > 0 else self.max_blocks_per_sm
        else:
            max_by_regs = self.max_blocks_per_sm

        active_blocks = min(max_by_warps, max_by_smem, max_by_regs, self.max_blocks_per_sm)
        active_warps = active_blocks * warps_per_block
        return min(100.0, max(0.0, active_warps / self.max_warps_per_sm * 100.0))

    # ── Source-Differentiated Stall Estimation ─────────────────────────────

    def _estimate_stalls(
        self,
        sp: SourceProfile,
        mem_pct: float,
        compute_pct: float,
        occupancy_pct: float,
    ) -> tuple[float, float, float, float]:
        """
        Estimate stalls from SOURCE PATTERNS + throughput.
        Different kernel implementations get DIFFERENT stall profiles.
        """
        barrier_points = sp.syncthreads + sp.syncwarp
        both_low = mem_pct < 30 and compute_pct < 30

        # ── Memory stall ──────────────────────────────────────────────────
        # Base: derived from throughput ratio
        if both_low:
            # Low utilization = latency-bound: high memory stalls (waiting for data)
            stall_memory = max(45.0, 75.0 - mem_pct)
        elif mem_pct > compute_pct and mem_pct > 30:
            stall_memory = min(80.0, (mem_pct - compute_pct) * 0.8 + 10.0)
        elif mem_pct > 50:
            stall_memory = mem_pct * 0.6
        else:
            stall_memory = max(10.0, 30.0 - compute_pct * 0.2)

        # Source adjustments: prefetch and vectorization REDUCE memory stalls
        if sp.has_prefetch:
            stall_memory *= 0.5  # cp.async/TMA hides latency
        if sp.has_vectorized_access:
            stall_memory *= 0.7  # fewer transactions, better coalescing
        if sp.has_smem_caching:
            stall_memory *= 0.8  # data reuse from shared memory
        if sp.ldg_hints > 0:
            stall_memory *= 0.9  # read-only cache path
        stall_memory = max(3.0, min(80.0, stall_memory))

        # ── Barrier stall ─────────────────────────────────────────────────
        # More __syncthreads = more barrier stalls
        if barrier_points == 0:
            stall_barrier = 1.0
        else:
            stall_barrier = min(55.0, barrier_points * 4.0)
            # Warp-level ops can replace some syncs
            if sp.has_warp_level_ops:
                stall_barrier *= 0.6

        # ── No-instruction stall ──────────────────────────────────────────
        if occupancy_pct < 50:
            stall_no_inst = (100.0 - occupancy_pct) * 0.4
        elif both_low:
            stall_no_inst = max(8.0, 25.0 - occupancy_pct * 0.1)
        else:
            stall_no_inst = max(2.0, (100.0 - occupancy_pct) * 0.15)
        # Loop unrolling increases ILP, reducing no-instruction stalls
        if sp.pragma_unroll > 0:
            stall_no_inst *= max(0.5, 1.0 - sp.pragma_unroll * 0.1)
        stall_no_inst = min(40.0, stall_no_inst)

        # ── MIO throttle ──────────────────────────────────────────────────
        # Shared memory contention
        if sp.smem_declarations > 0:
            stall_mio = min(30.0, sp.smem_declarations * 3.0 + sp.smem_reads * 0.3)
        else:
            stall_mio = max(1.0, mem_pct * 0.05)

        return stall_memory, stall_barrier, stall_no_inst, stall_mio

    # ── Compiler-Aware Stall Estimation ────────────────────────────────────

    def _estimate_stalls_from_compiler(
        self,
        cm: CompilerMetrics,
        sp: SourceProfile,
        mem_pct: float,
        compute_pct: float,
        occupancy_pct: float,
    ) -> tuple[float, float, float, float]:
        """Estimate stalls from COMPILER METRICS (real data) + source patterns.
        Uses SASS instruction counts and register pressure for differentiation."""

        has_sass = cm.sass_total_instructions > 0
        both_low = mem_pct < 30 and compute_pct < 30

        # ── Memory stall: based on spills + memory instruction ratio ──────
        if has_sass:
            # Spills are a strong signal of memory pressure
            spill_penalty = cm.spill_instruction_ratio * 5.0  # each % of spills = 5% stall
            mem_inst_ratio = cm.memory_instruction_ratio

            if both_low:
                stall_memory = max(35.0, 50.0 + spill_penalty - occupancy_pct * 0.2)
            elif mem_pct > 50:
                stall_memory = mem_pct * 0.7 + spill_penalty
            else:
                stall_memory = max(10.0, mem_inst_ratio * 0.8 + spill_penalty)

            # Vectorized loads reduce stalls
            if cm.vectorized_load_pct > 50:
                stall_memory *= max(0.5, 1.0 - cm.vectorized_load_pct / 200.0)
        else:
            # Fallback to source-based
            stall_memory, _, _, _ = self._estimate_stalls(
                sp, mem_pct, compute_pct, occupancy_pct
            )

        stall_memory = max(3.0, min(80.0, stall_memory))

        # ── Barrier stall: from SASS BAR count ────────────────────────────
        if has_sass:
            bar_ratio = (cm.sass_bar / cm.sass_total_instructions * 100) if cm.sass_total_instructions > 0 else 0
            stall_barrier = min(55.0, bar_ratio * 15.0 + cm.sass_bar * 2.0)
            # Warp shuffles replace barriers
            if cm.sass_shfl > 0:
                stall_barrier *= max(0.4, 1.0 - cm.sass_shfl / max(cm.sass_bar, 1) * 0.3)
        else:
            stall_barrier = min(55.0, sp.syncthreads * 4.0)
            if sp.has_warp_level_ops:
                stall_barrier *= 0.6

        stall_barrier = max(1.0, stall_barrier)

        # ── No-instruction stall: from occupancy + register pressure ──────
        if cm.registers_per_thread > 0:
            # High register count → fewer warps → more no-instruction stalls
            reg_pressure = max(0, cm.registers_per_thread - 32) * 1.5
            stall_no_inst = (100.0 - occupancy_pct) * 0.3 + reg_pressure
        elif occupancy_pct < 50:
            stall_no_inst = (100.0 - occupancy_pct) * 0.4
        else:
            stall_no_inst = max(2.0, (100.0 - occupancy_pct) * 0.15)
        stall_no_inst = max(2.0, min(40.0, stall_no_inst))

        # ── MIO throttle: from shared memory access density ───────────────
        if has_sass:
            smem_density = ((cm.sass_lds + cm.sass_sts) / cm.sass_total_instructions * 100) if cm.sass_total_instructions > 0 else 0
            stall_mio = min(30.0, smem_density * 3.0)
        else:
            stall_mio = min(30.0, sp.smem_declarations * 3.0 + sp.smem_reads * 0.3) if sp.smem_declarations > 0 else max(1.0, mem_pct * 0.05)

        return stall_memory, stall_barrier, stall_no_inst, stall_mio

    # ── Source-Adjusted Instruction Estimation ─────────────────────────────

    def _estimate_instructions(
        self, kernel_type: str, shape: tuple, sp: SourceProfile
    ) -> tuple[float, float, float, float]:
        """Estimate instructions adjusted by source patterns."""
        if kernel_type == "add_rmsnorm":
            rows, hidden = shape[0], shape[1]
            n = rows * hidden
            inst_load = float(n * 2 + hidden)
            inst_store = float(n + n // 2 + n // 16)
            inst_fp32 = float(n * 4)
            inst_fp16 = float(n * 2)
        elif kernel_type == "silu_mul":
            n = self._shape_to_n(shape)
            inst_load = float(n * 2)
            inst_store = float(n // 2 + n // 16)
            inst_fp32 = float(n * 4)
            inst_fp16 = float(n)
        elif kernel_type == "nvfp4_quantize":
            n = self._shape_to_n(shape)
            inst_load = float(n)
            inst_store = float(n // 2 + n // 16)
            inst_fp32 = float(n * 3)
            inst_fp16 = float(n)
        else:
            n = self._shape_to_n(shape)
            inst_load = float(n)
            inst_store = float(n)
            inst_fp32 = float(n * 2)
            inst_fp16 = 0.0

        # Vectorized loads/stores reduce instruction count (1 vector = 4-8 scalars)
        if sp.vectorized_loads > 0:
            vec_reduction = max(0.2, 1.0 - sp.vectorized_loads * 0.05)
            inst_load *= vec_reduction
        if sp.vectorized_stores > 0:
            inst_store *= max(0.3, 1.0 - sp.vectorized_stores * 0.05)

        # Warp shuffles add extra instructions
        inst_fp32 += sp.shfl_ops * 100
        inst_fp32 += sp.reduction_ops * 200

        return inst_load, inst_store, inst_fp32, inst_fp16

    # ── L2 Cache Estimation ────────────────────────────────────────────────

    def _estimate_l2_hit_rate(
        self, sp: SourceProfile, logical_bytes: float, effective_bytes: float,
        timing_sec: float,
    ) -> float:
        """Estimate L2 hit rate from source patterns + bandwidth analysis."""
        if timing_sec <= 0:
            return 0.0

        # Shared memory caching → higher L2 hit rate (less DRAM pressure)
        base_rate = 10.0
        if sp.has_smem_caching:
            base_rate += 25.0
        if sp.ldg_hints > 0:
            base_rate += 10.0

        # Small working set → high L2 hit rate
        if logical_bytes < self.l2_cache_bytes:
            base_rate += (1.0 - logical_bytes / self.l2_cache_bytes) * 30.0

        # Memory efficiency gap: if effective < logical, some reads hit cache
        if effective_bytes < logical_bytes:
            cache_ratio = 1.0 - effective_bytes / logical_bytes
            base_rate += cache_ratio * 20.0

        return min(95.0, max(5.0, base_rate))
