from feedback_bottleneck.config.args import RefinementPromptArgs
from feedback_bottleneck.llm.agent.refinement import (
    DEFAULT_TEACHER_SYSTEM_PROMPT,
    RefinementPromptSpec,
    _join_prompt_sections,
    render_ground_truth,
    render_student_response_instruction,
    render_task_block,
    render_teacher_output_instruction,
)

CODE_SYSTEM_PROMPT = "You are an expert competitive programmer."


def get_instruction_prompt(task: str = "default") -> str:
    del task
    return (
        "You are solving a programming task. Return a complete Python solution. "
        "When starter code is provided, follow its interface exactly."
    )


def _render_problem(obs) -> str:
    blocks = [f"Problem:\n{obs['obs']['problem']}"]
    starter_code = obs["obs"]["starter_code"]
    if starter_code:
        blocks.append(f"Starter code:\n{starter_code}")
    return _join_prompt_sections(*blocks)


def _render_execution_context(obs, teacher_execution_context_mode: str) -> str:
    latest_attempt = obs["obs"].get("latest_attempt") or {}
    latest_failure = obs["obs"].get("latest_failure") or {}

    if teacher_execution_context_mode == "none":
        return ""

    if teacher_execution_context_mode == "feedback":
        return f"""Here is the latest execution feedback:
{obs["obs"]["last_feedback"]}"""

    if teacher_execution_context_mode == "traceback":
        traceback_text = str(latest_failure.get("stderr", "")).strip()
        if traceback_text:
            return f"""Here is the latest traceback / stderr:
{traceback_text}"""
        return "No traceback / stderr is available from the latest attempt."

    if teacher_execution_context_mode == "structured":
        blocks = []
        if latest_attempt:
            blocks.append(
                "\n".join(
                    [
                        "Here is the latest sandbox result summary:",
                        f"- Passed {latest_attempt['num_passed']}/{latest_attempt['num_total']} tests.",
                        f"- Pass rate: {latest_attempt['pass_rate']:.2f}.",
                        f"- Timeout: {'yes' if latest_attempt['timeout'] else 'no'}.",
                    ]
                )
            )
        if latest_failure:
            failure_lines = ["Here are details from the first failing case:"]
            if latest_failure.get("error_type"):
                failure_lines.append(f"- Failure type: {latest_failure['error_type']}.")
            if latest_failure.get("input_preview"):
                failure_lines.append(f"- Input: {latest_failure['input_preview']}")
            if latest_failure.get("expected_preview"):
                failure_lines.append(f"- Expected: {latest_failure['expected_preview']}")
            if latest_failure.get("actual_preview"):
                failure_lines.append(f"- Actual: {latest_failure['actual_preview']}")
            if latest_failure.get("stderr"):
                failure_lines.append(f"- Traceback / stderr:\n{latest_failure['stderr']}")
            if latest_failure.get("stdout") and latest_failure.get("stdout") != latest_failure.get("actual_preview"):
                failure_lines.append(f"- Stdout:\n{latest_failure['stdout']}")
            blocks.append("\n".join(failure_lines))
        if not blocks:
            blocks.append("No structured execution details are available from the latest attempt.")
        return _join_prompt_sections(*blocks)

    raise ValueError(f"Unsupported teacher_execution_context_mode: {teacher_execution_context_mode}")


def _render_reference_context(obs, prompt_args: RefinementPromptArgs) -> str:
    return render_ground_truth(
        prompt_args,
        solution_text=f"""Here is a trusted reference solution:
```python
{obs["obs"]["reference_solution"]}
```""",
        none_message="You do not have access to a trusted reference solution.",
    )


def get_refinement_prompt_spec(
    task: str,
    *,
    prompt_args: RefinementPromptArgs,
) -> RefinementPromptSpec:
    del task

    def student_initial_prompt(obs) -> str:
        return _join_prompt_sections(
            render_task_block("competitive programming"),
            _render_problem(obs),
            "Solve the problem carefully.",
            render_student_response_instruction("Put the final code in a single ```python``` block."),
        )

    def student_refinement_prompt(obs, feedback: str) -> str:
        if prompt_args.feedback_mode == "feedback":
            return _join_prompt_sections(
                render_task_block("competitive programming"),
                f"Feedback:\n{feedback}",
                "Revise your previous solution so it uses the feedback.",
                render_student_response_instruction("Put the final code in a single ```python``` block."),
            )
        if prompt_args.feedback_mode == "no_feedback":
            return _join_prompt_sections(
                render_task_block("competitive programming"),
                "The previous attempt failed.",
                f"Latest execution feedback:\n{obs['obs']['last_feedback']}",
                "Revise your previous solution and produce a corrected solution.",
                render_student_response_instruction("Put the final code in a single ```python``` block."),
            )
        raise ValueError(f"Unsupported feedback_mode: {prompt_args.feedback_mode}")

    def teacher_initial_prompt(obs, previous_attempt: str) -> str:
        return _join_prompt_sections(
            render_task_block("competitive programming"),
            f"I am trying to solve this problem:\n{_render_problem(obs)}",
            f"Here is my current attempt:\n```python\n{previous_attempt}\n```",
            _render_execution_context(obs, prompt_args.teacher_execution_context_mode),
            _render_reference_context(obs, prompt_args),
            render_teacher_output_instruction(
                prompt_args,
                extra_rules=[
                    "Pinpoint the step in which I am making a mistake.",
                    "Provide the most informative piece of information for me to succeed on the next try, without telling me the final answer.",
                ],
            ),
        )

    def teacher_refinement_prompt(obs, previous_attempt: str) -> str:
        return _join_prompt_sections(
            render_task_block("competitive programming"),
            f"Here is my current attempt:\n```python\n{previous_attempt}\n```",
            _render_execution_context(obs, prompt_args.teacher_execution_context_mode),
            _render_reference_context(obs, prompt_args),
            render_teacher_output_instruction(
                prompt_args,
                extra_rules=[
                    "Pinpoint the step in which I am making a mistake.",
                    "Provide the most informative piece of information for me to succeed on the next try, without telling me the final answer.",
                ],
            ),
        )

    return RefinementPromptSpec(
        system_prompt=CODE_SYSTEM_PROMPT,
        student_initial_prompt=student_initial_prompt,
        student_refinement_prompt=student_refinement_prompt,
        teacher_initial_prompt=teacher_initial_prompt,
        teacher_refinement_prompt=teacher_refinement_prompt,
        teacher_system_prompt=DEFAULT_TEACHER_SYSTEM_PROMPT,
    )
