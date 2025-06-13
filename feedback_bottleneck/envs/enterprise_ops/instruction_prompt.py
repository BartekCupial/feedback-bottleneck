def get_instruction_prompt(task: str = "enterprise_ops") -> str:
    del task
    return (
        "Use the available EnterpriseOps MCP tools to complete the task. "
        "Tool actions must be JSON objects with `tool_name` and `arguments`; "
        "submit final work with a JSON object containing `final_response`."
    )
