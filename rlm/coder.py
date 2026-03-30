from __future__ import annotations


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

    parts = [
        f"You are the coder agent for branch \"{name}\".",
        f"Planner goal: {goal}",
        f"Required change: {change_summary}",
    ]

    if expected_signal:
        parts.append(f"Expected sandbox signal: {expected_signal}")
    if planner_notes:
        parts.append(f"Planner notes: {planner_notes}")
    if current_profile:
        parts.append(f"Current sandbox snapshot:\n{current_profile}")

    parts.append(f"Pinecone RAG context:\n{rag_context or 'No Pinecone context returned.'}")
    parts.append(f"Base kernel:\n```cuda\n{kernel_code}\n```")
    parts.append(launch_signature)
    parts.append(
        "Rules:\n"
        "- Implement this branch only.\n"
        "- Preserve correctness and the launch signature.\n"
        "- submit_kernel returns evaluator JSON, not prose. Use verdict, evidence, next_action, and rag.\n"
        "- Use search_pinecone when the attached RAG context is not enough.\n"
        "- Before submit_kernel, explain the bottleneck and the exact code change.\n"
        "- Then call submit_kernel with the complete .cu file."
    )

    return "\n\n".join(parts)
