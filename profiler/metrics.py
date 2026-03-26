"""
metrics.py — NCU metric extraction and naming conventions.
Maps human-readable metric names to NSight Compute metric IDs.
"""

from __future__ import annotations
from dataclasses import dataclass

# Key NCU metric IDs for B200 (sm_100a)
NCU_METRIC_IDS = {
    "mem_throughput_pct":     "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput_pct":      "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2_throughput_pct":      "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "compute_throughput_pct": "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "fp32_throughput_pct":    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm_occupancy":           "sm__warps_active.avg.pct_of_peak_sustained_active",
    "achieved_occupancy":     "sm__warps_active.avg.per_cycle_active",
    "stall_memory":           "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "stall_barrier":          "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "stall_no_instruction":   "smsp__warp_issue_stalled_no_instructions_per_warp_active.pct",
    "stall_mio_throttle":     "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "l2_hit_rate":            "lts__t_sector_hit_rate.pct",
    "l2_read_sectors":        "lts__t_sectors_op_read.sum",
    "l2_write_sectors":       "lts__t_sectors_op_write.sum",
    "dram_read_bytes":        "dram__bytes_read.sum",
    "dram_write_bytes":       "dram__bytes_write.sum",
    "dram_read_bw_gbps":      "dram__bytes_read.sum.per_second",
    "inst_executed":          "smsp__inst_executed.sum",
    "inst_fp32":              "smsp__inst_executed_pipe_fma.sum",
    "inst_fp16":              "smsp__inst_executed_pipe_fp16.sum",
    "inst_load":              "smsp__inst_executed_op_generic_ld.sum",
    "inst_store":             "smsp__inst_executed_op_generic_st.sum",
    "elapsed_cycles":         "sm__cycles_elapsed.sum",
    "active_cycles":          "sm__cycles_active.sum",
}

NCU_METRICS_QUERY = ",".join(NCU_METRIC_IDS.values())


@dataclass
class KernelMetrics:
    """Parsed profiler metrics for one kernel execution."""
    mem_throughput_pct:     float = 0.0
    l1_throughput_pct:      float = 0.0
    l2_throughput_pct:      float = 0.0
    compute_throughput_pct: float = 0.0
    fp32_throughput_pct:    float = 0.0
    sm_occupancy:           float = 0.0
    achieved_occupancy:     float = 0.0
    stall_memory:           float = 0.0
    stall_barrier:          float = 0.0
    stall_no_instruction:   float = 0.0
    stall_mio_throttle:     float = 0.0
    l2_hit_rate:            float = 0.0
    l2_read_sectors:        float = 0.0
    l2_write_sectors:       float = 0.0
    dram_read_bytes:        float = 0.0
    dram_write_bytes:       float = 0.0
    dram_read_bw_gbps:      float = 0.0
    inst_executed:          float = 0.0
    inst_fp32:              float = 0.0
    inst_fp16:              float = 0.0
    inst_load:              float = 0.0
    inst_store:             float = 0.0
    elapsed_cycles:         float = 0.0
    active_cycles:          float = 0.0
    duration_us:            float = 0.0
    speedup:                float = 1.0

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        # Include compiler metrics if available
        if hasattr(self, '_compiler_metrics') and self._compiler_metrics:
            d['_compiler'] = dataclasses.asdict(self._compiler_metrics)
        return d

    def summary_str(self) -> str:
        return (
            f"mem={self.mem_throughput_pct:.1f}% "
            f"compute={self.compute_throughput_pct:.1f}% "
            f"occupancy={self.sm_occupancy:.1f}% "
            f"stall_mem={self.stall_memory:.1f}% "
            f"stall_bar={self.stall_barrier:.1f}% "
            f"l2_hit={self.l2_hit_rate:.1f}% "
            f"speedup={self.speedup:.3f}x"
        )


@dataclass
class CompilerMetrics:
    """Metrics extracted from nvcc compiler output and SASS disassembly.
    These are EXACT values from the compiler — not heuristic estimates."""
    # From nvcc -Xptxas -v
    registers_per_thread: int = 0
    spill_stores_bytes: int = 0
    spill_loads_bytes: int = 0
    static_smem_bytes: int = 0
    cmem_bytes: int = 0
    stack_frame_bytes: int = 0

    # From cuobjdump -sass
    sass_total_instructions: int = 0
    sass_ldg_32: int = 0           # LDG.E (32-bit global load)
    sass_ldg_64: int = 0           # LDG.E.64 (64-bit)
    sass_ldg_128: int = 0          # LDG.E.128 (128-bit vectorized)
    sass_stg_32: int = 0           # STG.E
    sass_stg_64: int = 0           # STG.E.64
    sass_stg_128: int = 0          # STG.E.128
    sass_lds: int = 0              # LDS (shared memory load)
    sass_sts: int = 0              # STS (shared memory store)
    sass_ldl: int = 0              # LDL (local/spill load)
    sass_stl: int = 0              # STL (local/spill store)
    sass_ffma: int = 0             # FFMA (FP32 fused multiply-add)
    sass_hfma2: int = 0            # HFMA2 (FP16 FMA)
    sass_mufu: int = 0             # MUFU (special function unit)
    sass_fadd: int = 0             # FADD
    sass_fmul: int = 0             # FMUL
    sass_bar: int = 0              # BAR (barrier sync)
    sass_shfl: int = 0             # SHFL (warp shuffle)
    sass_bra: int = 0              # BRA (branch)

    @property
    def has_spills(self) -> bool:
        return self.spill_stores_bytes > 0 or self.spill_loads_bytes > 0

    @property
    def vectorized_load_pct(self) -> float:
        total_ldg = self.sass_ldg_32 + self.sass_ldg_64 + self.sass_ldg_128
        return (self.sass_ldg_128 / total_ldg * 100) if total_ldg > 0 else 0.0

    @property
    def memory_instruction_ratio(self) -> float:
        mem = (self.sass_ldg_32 + self.sass_ldg_64 + self.sass_ldg_128 +
               self.sass_stg_32 + self.sass_stg_64 + self.sass_stg_128 +
               self.sass_lds + self.sass_sts)
        return (mem / self.sass_total_instructions * 100) if self.sass_total_instructions > 0 else 0.0

    @property
    def spill_instruction_ratio(self) -> float:
        spills = self.sass_ldl + self.sass_stl
        return (spills / self.sass_total_instructions * 100) if self.sass_total_instructions > 0 else 0.0

    def summary_str(self) -> str:
        parts = [f"regs={self.registers_per_thread}"]
        if self.has_spills:
            parts.append(f"spill_ld={self.spill_loads_bytes}B spill_st={self.spill_stores_bytes}B")
        parts.append(f"smem={self.static_smem_bytes}B")
        if self.sass_total_instructions > 0:
            parts.append(f"sass_insts={self.sass_total_instructions}")
            parts.append(f"vec_ld%={self.vectorized_load_pct:.0f}")
            if self.sass_ldl + self.sass_stl > 0:
                parts.append(f"spill_insts={self.sass_ldl+self.sass_stl}")
        return " ".join(parts)


def parse_ncu_csv_line(metric_id: str, value_str: str) -> float:
    value_str = value_str.strip().replace(",", "")
    for suffix in [" %", " GB/s", " TB/s", " MB/s", " GHz", " ns", " us", " ms"]:
        if value_str.endswith(suffix):
            value_str = value_str[:-len(suffix)]
    try:
        return float(value_str)
    except ValueError:
        return 0.0


def metrics_from_dict(d: dict) -> KernelMetrics:
    m = KernelMetrics()
    for field_name, value in d.items():
        if hasattr(m, field_name):
            setattr(m, field_name, float(value))
    return m
