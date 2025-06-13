from __future__ import annotations

from typing import Any

from feedback_bottleneck.envs.enterprise_ops.benchmark.mcp_client import MCPClient
from feedback_bottleneck.envs.enterprise_ops.benchmark.models import VerifierConfig
from feedback_bottleneck.envs.enterprise_ops.benchmark.verifier import VerifierEngine
from feedback_bottleneck.envs.enterprise_ops.tasks import EnterpriseOpsTask


async def run_enterprise_ops_verifiers(
    task: EnterpriseOpsTask,
    mcp_clients: dict[str, MCPClient],
    database_ids: dict[str, str],
    final_response: str,
    attempts: list[dict[str, Any]],
) -> tuple[float, str, dict[str, Any], dict[str, Any]]:
    model_response = {
        "content": final_response,
        "tool_calls": [
            {
                "gym_name": attempt["gym_name"],
                "name": attempt["tool_name"],
                "args": attempt["arguments"],
            }
            for attempt in attempts
            if attempt.get("kind") == "tool_call"
        ],
    }
    engine = VerifierEngine(mcp_clients)
    servers_by_name = {server.gym_name: server for server in task.servers}
    results = {}

    for idx, verifier_config in enumerate(task.verifiers):
        verifier = VerifierConfig(**verifier_config)
        if verifier.verifier_type == "response_check":
            raise ValueError("EnterpriseOps response_check verifier is not wired into this env yet.")

        gym_name = verifier.gym_name
        if gym_name is None:
            if len(task.servers) != 1:
                raise ValueError(
                    f"EnterpriseOps verifier {verifier.name or idx + 1} has no gym_name "
                    f"for multi-server task {task.task_id}."
                )
            gym_name = task.servers[0].gym_name

        results[verifier.name or f"verifier_{idx + 1}"] = await engine.execute_verifier(
            verifier,
            model_response,
            database_ids[gym_name],
            servers_by_name[gym_name].context,
            gym_name=gym_name,
        )

    total = len(results)
    passed = sum(1 for result in results.values() if result["passed"])
    failed = total - passed
    summary = {
        "enterprise_ops_total_verifiers": total,
        "enterprise_ops_passed_verifiers": passed,
        "enterprise_ops_failed_verifiers": failed,
        "enterprise_ops_verifier_pass_rate": passed / total,
        "enterprise_ops_overall_success": float(failed == 0),
    }

    failed_names = [name for name, result in results.items() if not result["passed"]]
    feedback = f"EnterpriseOps verification passed {passed}/{total} verifiers."
    if failed_names:
        feedback += f" Failed verifiers: {', '.join(failed_names)}."
    return float(failed == 0), feedback, results, summary
