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
    bottleneck = plan_branch.get("bottleneck", "")
    rationale = plan_branch.get("rationale", "")
    risk = plan_branch.get("risk", "")
    evidence = plan_branch.get("evidence", []) or []

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
    parts.append(
        "Rules:\n"
        "- Implement this branch only.\n"
        "- Planner owns strategy selection. Do not change branch family, optimization surface, or overall direction.\n"
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
