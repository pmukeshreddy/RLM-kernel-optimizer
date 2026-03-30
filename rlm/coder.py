from __future__ import annotations


def _infer_kernel_type(launch_signature: str, kernel_code: str) -> str:
    haystack = f"{launch_signature}\n{kernel_code}"
    if "launch_fused_add_rmsnorm_nvfp4" in haystack:
        return "add_rmsnorm"
    if "launch_silu_mul_fp4quant" in haystack:
        return "silu_mul"
    if "launch_nvfp4_quantize_bf16" in haystack:
        return "nvfp4_quantize"
    return "unknown"


def _branch_mentions(text: str, *terms: str) -> bool:
    norm = text.lower()
    return any(term in norm for term in terms)


def build_coder_prompt(
    plan_branch: dict,
    kernel_code: str,
    launch_signature: str,
    rag_context: str,
    current_profile: str = "",
) -> str:
    name = plan_branch.get("name", "unnamed_branch")
    goal = plan_branch.get("goal") or plan_branch.get("change_summary") or plan_branch.get("what", "")
    change_summary = plan_branch.get("change_summary") or plan_branch.get("what") or goal
    expected_signal = plan_branch.get("expected_signal", "")
    planner_notes = plan_branch.get("planner_notes", "")
    bottleneck = plan_branch.get("bottleneck", "")
    rationale = plan_branch.get("rationale", "")
    risk = plan_branch.get("risk", "")
    evidence = plan_branch.get("evidence", []) or []
    kernel_type = _infer_kernel_type(launch_signature, kernel_code)
    branch_text = " ".join(
        str(item)
        for item in (
            name,
            goal,
            change_summary,
            planner_notes,
            rationale,
            risk,
            " ".join(str(item) for item in evidence),
        )
        if item
    ).lower()

    parts = [
        f"You are the coder agent for branch \"{name}\".",
        f"Planner goal: {goal}",
        f"Required change: {change_summary}",
    ]

    if bottleneck:
        parts.append(f"Observed concern: {bottleneck}")
    if expected_signal:
        parts.append(f"Expected sandbox signal: {expected_signal}")
    if planner_notes:
        parts.append(f"Planner notes: {planner_notes}")
    if rationale:
        parts.append(f"Planner rationale: {rationale}")
    if risk:
        parts.append(f"Primary risk: {risk}")
    if evidence:
        parts.append("Planner evidence:\n- " + "\n- ".join(str(item) for item in evidence[:4]))
    if current_profile:
        parts.append(f"Current sandbox snapshot:\n{current_profile}")

    parts.append(f"Pinecone RAG context:\n{rag_context or 'No Pinecone context returned.'}")
    parts.append(f"Base kernel:\n```cuda\n{kernel_code}\n```")
    parts.append(launch_signature)

    kernel_specific_rules = []
    if kernel_type == "add_rmsnorm":
        kernel_specific_rules.extend([
            "For add+rmsnorm+fp4 on shape 128x2048, treat the Phase-2 residual_out reread as a primary cost center.",
            "Prefer project helpers from kernels/common/nvfp4_utils.cuh (for example pack_fp4_pair / quantize_block_nvfp4) over re-implementing a scalar branch chain.",
            "Preserve the working occupancy regime. The strong working path is around 32 registers/thread and 100% occupancy.",
            "Hard guard: if your first submit on this 128x2048 kernel shows registers above 32 and speedup below 1.05x, you MUST revert in the next turn. Do not iterate further on a high-register path.",
        ])
        if not _branch_mentions(branch_text, "warp", "shuffle", "reduction", "shfl", "syncthreads"):
            kernel_specific_rules.append(
                "This branch is NOT a reduction branch. Preserve the existing reduction structure instead of sneaking in warp-reduction changes."
            )
        if _branch_mentions(branch_text, "fuse", "single pass", "single-pass", "reread", "re-read", "smem cache"):
            kernel_specific_rules.extend([
                "This branch should eliminate the second global-memory read of residual_out.",
                "Each thread owns exactly 8 elements (2048/256). If you cache them, use a float reg[8] budget consciously.",
                "A float reg[8] cache costs about 8 registers. Starting from a ~32-register working path, that puts you near ~40 registers, which may drop occupancy sharply.",
                "If you use reg[8], keep every other change minimal and avoid adding extra arrays or shared-memory staging unless absolutely necessary.",
                "Abort condition for the single-pass path: if registers reach 40 and speedup stays below 1.05x, revert to the 32-register version immediately and do not continue refining the high-register path.",
            ])
        if _branch_mentions(branch_text, "fp4", "intrinsic", "pack", "quant"):
            kernel_specific_rules.append(
                "Do not leave the scalar float_to_nvfp4 if/else chain as the hot-path encoder if a project helper or hardware intrinsic path can replace it."
            )

    if kernel_specific_rules:
        parts.append("Kernel-specific rules:\n- " + "\n- ".join(kernel_specific_rules))

    parts.append(
        "Rules:\n"
        "- Implement this branch only.\n"
        "- Planner owns strategy selection. Do not change branch family, optimization surface, or overall direction.\n"
        "- Do not silently add a second optimization surface. If the branch is about vectorized loads, do not also change reduction or quantization strategy unless the branch explicitly says so.\n"
        "- Preserve correctness and the launch signature.\n"
        "- submit_kernel returns evaluator JSON, not prose. Use observations, hypothesis_test, next_action, memory, uncertainty, and rag.\n"
        "- Use the attached Pinecone RAG context as the planner-approved reference set. Do not start a new Pinecone search or invent a new strategy.\n"
        "- Treat observations as ground truth. Treat hypothesis_test.status as the verdict on your last idea.\n"
        "- If the attached RAG context shows a strong production pattern, adapt that pattern as faithfully as possible within this branch.\n"
        "- Make one local experiment at a time. Follow next_action.success_criteria and next_action.abort_if.\n"
        "- If the branch seems weak, keep the adaptation minimal and let the sandbox result send control back to the planner.\n"
        "- Before submit_kernel, explain what the latest result confirmed or left uncertain and the exact code change.\n"
        "- Then call submit_kernel with the complete .cu file."
    )

    return "\n\n".join(parts)
