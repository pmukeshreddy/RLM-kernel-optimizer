"""
test_llm_picks.py — Test whether the LLM can pick the right strategies for each kernel.

Sends each kernel's source code + full strategy menu to the LLM and prints what it picks.
No compilation, no profiling — just strategy selection.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 test_llm_picks.py                          # default: haiku
    python3 test_llm_picks.py --model opus              # test with opus
    python3 test_llm_picks.py --model sonnet             # test with sonnet
    python3 test_llm_picks.py --model all                # compare all 3
"""

import argparse
import json
import re
import time
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).parent

# ── Model IDs ────────────────────────────────────────────────────────────────

MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

# ── Load strategy bank ───────────────────────────────────────────────────────

# We inline this to avoid circular import issues
STRATEGY_DESCRIPTIONS = {
    "vectorize_loads": {
        "desc": "Replace scalar loads with 128-bit float4/uint4 vectorized transactions",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "tma_prefetch": {
        "desc": "Use Blackwell TMA engine for async bulk copy with double buffering. "
                "Best for 2D/3D tiled access patterns, NOT simple linear access",
        "applicable": [],
    },
    "warp_reduction": {
        "desc": "Replace __syncthreads block reductions with __shfl_xor_sync warp shuffles",
        "applicable": ["add_rmsnorm"],
    },
    "fuse_passes": {
        "desc": "Combine multiple global memory passes into a single kernel loop. "
                "Only useful when kernel reads the same data in multiple passes",
        "applicable": ["add_rmsnorm"],
    },
    "register_tiling": {
        "desc": "Process multiple elements per thread using register arrays for ILP",
        "applicable": ["add_rmsnorm", "silu_mul"],
    },
    "async_pipeline": {
        "desc": "Overlap memory and compute using cp.async with double buffering. "
                "Best for kernels with loop-carried dependencies to overlap",
        "applicable": [],
    },
    "fp4_lut": {
        "desc": "Replace arithmetic FP4 quantization with a lookup table. "
                "FP4 has only 16 possible output values — a LUT eliminates all quantize math",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "fast_math_expf": {
        "desc": "Replace slow expf/logf libcalls (~20 cycles) with hardware SFU intrinsics "
                "__expf/__logf (~4 cycles). Precision loss invisible when output goes to FP4",
        "applicable": ["silu_mul"],
    },
    "thread_coarsening": {
        "desc": "Increase work per thread (multiple quant blocks or rows per thread). "
                "Amortizes thread launch overhead when per-thread work is too small",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "ldg_readonly": {
        "desc": "Route read-only inputs through L1 texture cache using __ldg(). "
                "Uses a separate cache path, freeing normal L1 for read-write data",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "vectorized_stores": {
        "desc": "Replace byte-by-byte packed FP4 output stores with uint2/uint4 writes. "
                "Reduces store instruction count by 8x",
        "applicable": ["nvfp4_quantize", "silu_mul", "add_rmsnorm"],
    },
}

# ── Kernel sources ────────────────────────────────────────────────────────────

KERNELS = {
    "add_rmsnorm": PROJECT_ROOT / "kernels" / "reference" / "add_rmsnorm.cu",
    "silu_mul":    PROJECT_ROOT / "kernels" / "reference" / "silu_mul.cu",
    "nvfp4_quantize": PROJECT_ROOT / "kernels" / "reference" / "nvfp4_quantize.cu",
}

# ── What a human expert would pick (ground truth for comparison) ──────────────

EXPERT_PICKS = {
    "add_rmsnorm":    ["fuse_passes", "vectorize_loads", "fp4_lut", "thread_coarsening"],
    "silu_mul":       ["vectorize_loads", "fast_math_expf", "fp4_lut", "thread_coarsening"],
    "nvfp4_quantize": ["fp4_lut", "vectorize_loads", "thread_coarsening", "ldg_readonly"],
}


def build_strategy_menu(kernel_type: str) -> str:
    """Build a formatted strategy menu showing all strategies."""
    lines = []
    for name, info in STRATEGY_DESCRIPTIONS.items():
        applicable = info["applicable"]
        applies = "YES" if (not applicable or kernel_type in applicable) else "NO (not applicable to this kernel type)"
        lines.append(f"  - {name}: {info['desc']}")
        lines.append(f"    Applicable to this kernel: {applies}")
    return "\n".join(lines)


def build_prompt(kernel_src: str, kernel_type: str) -> str:
    menu = build_strategy_menu(kernel_type)
    return f"""\
Analyze this CUDA kernel and select exactly 4 optimization strategies that would
have the HIGHEST IMPACT for this specific kernel.

Kernel type: {kernel_type}
Target: NVIDIA B200 (Blackwell, sm_100a)

```cuda
{kernel_src}
```

Available strategies:
{menu}

Think about:
1. What does this kernel actually do? (elementwise? reduction? multi-pass?)
2. Where is it spending time? (memory loads? compute? synchronization?)
3. Which strategies match the kernel's actual access pattern?

Return ONLY a JSON array of exactly 4 strategy names, most impactful first:
["strategy1", "strategy2", "strategy3", "strategy4"]
"""


def call_llm(client: anthropic.Anthropic, prompt: str, model_id: str) -> tuple:
    """Call the LLM and return (parsed_strategies, raw_response, cost, latency)."""
    t0 = time.time()
    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - t0
    text = response.content[0].text
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens

    # Parse JSON array from response
    parsed = []
    json_match = re.search(r'\[.*?\]', text, re.DOTALL)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            parsed = [s for s in raw if isinstance(s, str)]
        except (json.JSONDecodeError, TypeError):
            pass

    return parsed, text, tokens_in, tokens_out, latency


def score_picks(picks: list, expert: list) -> dict:
    """Score LLM picks against expert ground truth."""
    pick_set = set(picks[:4])
    expert_set = set(expert)
    overlap = pick_set & expert_set
    # Check if any picks are inapplicable (from STRATEGY_DESCRIPTIONS)
    return {
        "overlap": len(overlap),
        "total": len(expert_set),
        "matching": sorted(overlap),
        "missed": sorted(expert_set - pick_set),
        "extra": sorted(pick_set - expert_set),
        "score_pct": len(overlap) / len(expert_set) * 100,
    }


def run_test(client: anthropic.Anthropic, model_name: str, model_id: str):
    """Run the strategy selection test for all 3 kernels."""
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name} ({model_id})")
    print(f"{'='*70}")

    total_score = 0
    total_possible = 0
    total_cost = 0.0

    pricing = {
        "haiku":  {"in": 0.25, "out": 1.25},
        "sonnet": {"in": 3.0,  "out": 15.0},
        "opus":   {"in": 15.0, "out": 75.0},
    }
    price = pricing.get(model_name, {"in": 3.0, "out": 15.0})

    for kernel_type, src_path in KERNELS.items():
        kernel_src = src_path.read_text()
        prompt = build_prompt(kernel_src, kernel_type)

        print(f"\n  --- {kernel_type} ---")
        picks, raw, tok_in, tok_out, latency = call_llm(client, prompt, model_id)
        cost = (tok_in * price["in"] + tok_out * price["out"]) / 1_000_000
        total_cost += cost

        expert = EXPERT_PICKS[kernel_type]
        result = score_picks(picks, expert)
        total_score += result["overlap"]
        total_possible += result["total"]

        print(f"  LLM picked:    {picks}")
        print(f"  Expert ideal:  {expert}")
        print(f"  Match:         {result['overlap']}/4 ({result['score_pct']:.0f}%)")
        if result["matching"]:
            print(f"    correct:     {result['matching']}")
        if result["missed"]:
            print(f"    missed:      {result['missed']}")
        if result["extra"]:
            print(f"    extra:       {result['extra']}")
        print(f"  Tokens:        in={tok_in} out={tok_out}")
        print(f"  Cost:          ${cost:.4f}")
        print(f"  Latency:       {latency:.1f}s")

        # Show raw response if parsing failed
        if not picks:
            print(f"  RAW RESPONSE:  {raw[:200]}")

    print(f"\n  {'─'*50}")
    print(f"  TOTAL SCORE:   {total_score}/{total_possible} "
          f"({total_score/total_possible*100:.0f}%)")
    print(f"  TOTAL COST:    ${total_cost:.4f}")
    print(f"  {'─'*50}")
    return total_score, total_possible, total_cost


def main():
    parser = argparse.ArgumentParser(description="Test LLM strategy selection")
    parser.add_argument("--model", default="haiku",
                        choices=["haiku", "sonnet", "opus", "all"],
                        help="Which model to test (default: haiku)")
    args = parser.parse_args()

    client = anthropic.Anthropic()

    if args.model == "all":
        results = {}
        for name, model_id in MODELS.items():
            score, total, cost = run_test(client, name, model_id)
            results[name] = {"score": score, "total": total, "cost": cost}

        print(f"\n{'='*70}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*70}")
        for name, r in results.items():
            pct = r["score"] / r["total"] * 100
            print(f"  {name:8s}  {r['score']}/{r['total']} ({pct:.0f}%)  "
                  f"cost=${r['cost']:.4f}")
    else:
        model_id = MODELS[args.model]
        run_test(client, args.model, model_id)


if __name__ == "__main__":
    main()
