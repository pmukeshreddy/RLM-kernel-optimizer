"""
metrics.py — Kernel metric definitions and parsing utilities.
"""

from __future__ import annotations
from dataclasses import dataclass


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
            f"timing={self.duration_us:.1f}us "
            f"occupancy={self.sm_occupancy:.1f}% "
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


def metrics_from_dict(d: dict) -> KernelMetrics:
    m = KernelMetrics()
    for field_name, value in d.items():
        if field_name == '_compiler':
            continue  # handled below
        if hasattr(m, field_name):
            setattr(m, field_name, float(value))
    # Reconstruct compiler metrics so to_dict() preserves _compiler
    compiler_data = d.get('_compiler')
    if compiler_data:
        cm = CompilerMetrics()
        for k, v in compiler_data.items():
            if hasattr(cm, k):
                setattr(cm, k, int(v) if isinstance(v, (int, float)) else v)
        m._compiler_metrics = cm
    return m
