"""
test_llm_codegen.py — Test if LLMs can write compilable, optimized CUDA kernels.

For each kernel, asks the LLM to apply a specific optimization and tries to compile.
Measures: compile pass rate, code extraction success, and latency.

Usage:
    python3 test_llm_codegen.py --model sonnet
    python3 test_llm_codegen.py --model all
    python3 test_llm_codegen.py --model opus --kernel silu_mul   # single kernel
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).parent

MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

PRICING = {
    "haiku":  {"in": 0.25, "out": 1.25},
    "sonnet": {"in": 3.0,  "out": 15.0},
    "opus":   {"in": 15.0, "out": 75.0},
}

# ── Kernel sources ────────────────────────────────────────────────────────────

KERNELS = {
    "add_rmsnorm":    PROJECT_ROOT / "kernels" / "reference" / "add_rmsnorm.cu",
    "silu_mul":       PROJECT_ROOT / "kernels" / "reference" / "silu_mul.cu",
    "nvfp4_quantize": PROJECT_ROOT / "kernels" / "reference" / "nvfp4_quantize.cu",
}

COMMON_DIR = PROJECT_ROOT / "kernels" / "common"

# ── Optimizations to test (from Sonnet freeform results — the best picks) ────

TASKS = {
    "add_rmsnorm": [
        {
            "name": "vectorized_loads",
            "instruction": (
                "Replace all scalar bfloat16 loads with 128-bit vectorized loads. "
                "Use uint4 loads and reinterpret as __nv_bfloat162 pairs to load "
                "8 bf16 elements per transaction. Apply to input, residual, weight, "
                "and residual_out arrays in both Phase 1 and Phase 2 loops. "
                "Adjust loop stride accordingly (tid*8 instead of tid)."
            ),
        },
        {
            "name": "fuse_passes",
            "instruction": (
                "Eliminate the global memory round-trip between Phase 1 and Phase 2. "
                "Currently Phase 1 writes residual_out[base+i] to HBM, then Phase 2 "
                "re-reads it. Instead, keep the added values (a+r) in registers or "
                "shared memory during Phase 1, compute the RMS inverse, then immediately "
                "normalize and quantize without re-reading from global memory."
            ),
        },
        {
            "name": "vectorized_stores",
            "instruction": (
                "Replace the byte-by-byte quant_out writes with vectorized stores. "
                "The loop `for (j=0; j<8; ++j) quant_out[base+j] = packed_out[j]` "
                "does 8 separate byte stores. Pack the 8 bytes into a uint2 and write "
                "with a single 64-bit store: `*(uint2*)&quant_out[base] = packed_val`. "
                "Ensure 8-byte alignment."
            ),
        },
    ],
    "silu_mul": [
        {
            "name": "vectorized_loads",
            "instruction": (
                "Replace scalar bfloat16 loads of gate and up arrays with 128-bit "
                "vectorized loads. Use uint4 to load 8 bf16 values at once from each "
                "array. Reinterpret as __nv_bfloat162 for conversion to float. "
                "Each thread should still process one 16-element NVFP4 block but "
                "load in two 8-element vector transactions."
            ),
        },
        {
            "name": "fast_math_expf",
            "instruction": (
                "Replace the slow expf() call in silu_f32() with __expf() hardware "
                "intrinsic. Also replace the reciprocal with __frcp_rn(). The optimized "
                "SiLU becomes: x * __frcp_rn(1.0f + __expf(-x)). This uses the SFU "
                "unit (~4 cycles vs ~20 for software expf). Precision loss is invisible "
                "since output goes to FP4."
            ),
        },
        {
            "name": "thread_coarsening",
            "instruction": (
                "Have each thread process 4 NVFP4 blocks (64 elements) instead of 1. "
                "Use a loop over 4 blocks per thread. Divide the grid size by 4. "
                "Keep all values in registers. Add bounds checking for the last "
                "iteration when num_quant_blocks is not divisible by 4."
            ),
        },
    ],
    "nvfp4_quantize": [
        {
            "name": "vectorized_loads",
            "instruction": (
                "Replace the scalar bfloat16 loads with 128-bit vectorized loads. "
                "Use uint4 to load 8 bf16 values at once (16 bytes). Each NVFP4 block "
                "is 16 elements, so 2 vector loads per block. Reinterpret the uint4 "
                "components as __nv_bfloat162 pairs for conversion to float."
            ),
        },
        {
            "name": "vectorized_stores",
            "instruction": (
                "Replace byte-by-byte packed output stores with a single 64-bit store. "
                "The 8 packed bytes per NVFP4 block should be accumulated into a uint2 "
                "and written with: *(uint2*)&packed[packed_base] = packed_val. This "
                "replaces 8 byte stores with 1 instruction."
            ),
        },
        {
            "name": "thread_coarsening",
            "instruction": (
                "Have each thread process 4 quantization blocks instead of 1. "
                "Loop over 4 blocks sequentially, keeping all values in registers. "
                "Reduce grid size by 4x. Add bounds check: if (block_id + i*stride >= "
                "num_blocks) break. Use #pragma unroll on the 4-iteration outer loop."
            ),
        },
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def expand_includes(src: str, src_path: Path) -> str:
    """Expand local #include directives so the LLM sees helper functions."""
    lines = src.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#include "') and stripped.endswith('"'):
            rel_path = stripped[len('#include "'):-1]
            header = src_path.parent / rel_path
            if header.exists():
                result.append(f"// === expanded from {rel_path} ===")
                result.append(header.read_text())
                result.append(f"// === end {rel_path} ===")
                continue
        result.append(line)
    return "\n".join(result)


def extract_cuda_code(text: str) -> str:
    """Extract CUDA code from LLM response."""
    for pattern in [r"```cuda\s*\n(.*?)```", r"```cpp\s*\n(.*?)```",
                    r"```c\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    if "__global__" in text or "#include" in text:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("#include") or s.startswith("__global__") or s.startswith("//"):
                return "\n".join(lines[i:]).strip()
    return ""


def try_compile(cuda_code: str, kernel_type: str) -> tuple:
    """Try to compile CUDA code with nvcc. Returns (success, error_msg)."""
    ncu_path = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = os.path.join(ncu_path, "bin", "nvcc")
    if not os.path.exists(nvcc):
        nvcc = "nvcc"  # hope it's on PATH

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = os.path.join(tmpdir, f"{kernel_type}_optimized.cu")
        obj_file = os.path.join(tmpdir, f"{kernel_type}_optimized.o")

        # Write the code, with include path set to common/
        with open(src_file, "w") as f:
            f.write(cuda_code)

        try:
            result = subprocess.run(
                [nvcc, "-c", "-O3", "--std=c++17",
                 "-arch=sm_100a",
                 f"-I{COMMON_DIR}",
                 f"-I{PROJECT_ROOT / 'kernels' / 'reference'}",
                 "-o", obj_file, src_file],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                return True, ""
            else:
                # Extract the key error lines
                err = result.stderr
                err_lines = [l for l in err.split("\n") if "error" in l.lower()]
                return False, "\n".join(err_lines[:5]) if err_lines else err[:300]
        except FileNotFoundError:
            return False, "nvcc not found"
        except subprocess.TimeoutExpired:
            return False, "compilation timed out (60s)"


def build_codegen_prompt(kernel_src_expanded: str, task: dict) -> str:
    """Build a prompt asking the LLM to apply a specific optimization."""
    return f"""\
You are an expert CUDA kernel optimizer targeting NVIDIA B200 (sm_100a, Blackwell).

Apply this optimization to the kernel below:

## Optimization: {task['name']}
{task['instruction']}

## Full kernel source (headers expanded inline between === markers):
```cuda
{kernel_src_expanded}
```

CRITICAL RULES:
1. Return the COMPLETE .cu file in a single ```cuda code block
2. Keep all #includes (use the original #include directives, NOT the expanded content)
3. Keep ALL kernel functions and the launch_* wrapper function
4. Do NOT change the launch_* function signature
5. Output must match reference within atol=1e-2
6. No explanations — just the code block
"""


# ── Main test ─────────────────────────────────────────────────────────────────

def run_test(client: anthropic.Anthropic, model_name: str, model_id: str,
             kernel_filter: str = None):
    print(f"\n{'='*70}")
    print(f"  CODEGEN TEST | MODEL: {model_name}")
    print(f"{'='*70}")

    price = PRICING.get(model_name, {"in": 3.0, "out": 15.0})
    total_cost = 0.0
    total_tasks = 0
    extract_pass = 0
    compile_pass = 0
    results = []

    for kernel_type, tasks in TASKS.items():
        if kernel_filter and kernel_type != kernel_filter:
            continue

        src_path = KERNELS[kernel_type]
        kernel_src_raw = src_path.read_text()
        kernel_src_expanded = expand_includes(kernel_src_raw, src_path)

        print(f"\n  ── {kernel_type} ──")

        for task in tasks:
            total_tasks += 1
            prompt = build_codegen_prompt(kernel_src_expanded, task)

            t0 = time.time()
            response = client.messages.create(
                model=model_id,
                max_tokens=8192,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            latency = time.time() - t0
            text = response.content[0].text
            tok_in = response.usage.input_tokens
            tok_out = response.usage.output_tokens
            cost = (tok_in * price["in"] + tok_out * price["out"]) / 1_000_000
            total_cost += cost

            # Extract code
            code = extract_cuda_code(text)
            has_code = bool(code)
            if has_code:
                extract_pass += 1

            # Try compile
            compiled = False
            err_msg = ""
            if has_code:
                compiled, err_msg = try_compile(code, kernel_type)
                if compiled:
                    compile_pass += 1

            # Status symbol
            if compiled:
                status = "COMPILE OK"
            elif has_code:
                status = "COMPILE FAIL"
            else:
                status = "NO CODE"

            print(f"    {task['name']:30s}  {status:15s}  "
                  f"tok={tok_out:4d}  ${cost:.4f}  {latency:.1f}s")
            if err_msg:
                # Show first error line
                first_err = err_msg.split("\n")[0][:100]
                print(f"      err: {first_err}")

            results.append({
                "kernel": kernel_type,
                "strategy": task["name"],
                "model": model_name,
                "extracted": has_code,
                "compiled": compiled,
                "tokens_out": tok_out,
                "cost": cost,
                "latency": latency,
                "error": err_msg[:200] if err_msg else "",
            })

    # Summary
    print(f"\n  {'─'*50}")
    print(f"  CODE EXTRACTION:  {extract_pass}/{total_tasks} "
          f"({extract_pass/total_tasks*100:.0f}%)")
    print(f"  COMPILE PASS@1:   {compile_pass}/{total_tasks} "
          f"({compile_pass/total_tasks*100:.0f}%)")
    print(f"  TOTAL COST:       ${total_cost:.4f}")
    print(f"  {'─'*50}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test LLM CUDA code generation")
    parser.add_argument("--model", default="sonnet",
                        choices=["haiku", "sonnet", "opus", "all"])
    parser.add_argument("--kernel", default=None,
                        choices=["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
                        help="Test only one kernel type")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    all_results = []

    if args.model == "all":
        for name, model_id in MODELS.items():
            results = run_test(client, name, model_id, args.kernel)
            all_results.extend(results)

        # Comparison
        print(f"\n{'='*70}")
        print(f"  COMPARISON")
        print(f"{'='*70}")
        for name in MODELS:
            mine = [r for r in all_results if r["model"] == name]
            compiled = sum(1 for r in mine if r["compiled"])
            total = len(mine)
            cost = sum(r["cost"] for r in mine)
            print(f"  {name:8s}  compile={compiled}/{total} "
                  f"({compiled/total*100:.0f}%)  cost=${cost:.4f}")
    else:
        model_id = MODELS[args.model]
        run_test(client, args.model, model_id, args.kernel)


if __name__ == "__main__":
    main()
