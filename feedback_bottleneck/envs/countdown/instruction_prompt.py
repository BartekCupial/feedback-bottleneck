from feedback_bottleneck.config.args import RefinementPromptArgs
from feedback_bottleneck.llm.agent.refinement import (
    DEFAULT_TEACHER_SYSTEM_PROMPT,
    RefinementPromptSpec,
    _join_prompt_sections,
    render_student_response_instruction,
    render_task_block,
    render_teacher_output_instruction,
)

COUNTDOWN_SYSTEM_PROMPT = (
    "You are solving Countdown arithmetic tasks. "
    "Think through the arithmetic carefully and return your final expression inside <answer>...</answer>."
)


def get_instruction_prompt(task: str):
    del task
    return (
        "Given a target and a set of numbers, produce an arithmetic expression that reaches the target. "
        "Use only +, -, *, / and do not use any number more times than provided."
    )


def get_refinement_prompt_spec(task: str, *, prompt_args: RefinementPromptArgs) -> RefinementPromptSpec:
    del task

    def student_initial_prompt(obs) -> str:
        problem = obs["obs"]["problem"]
        return _join_prompt_sections(
            render_task_block("countdown arithmetic"),
            f"Problem:\n{problem}",
            render_student_response_instruction(
                "Your final expression must appear exactly once in <answer>...</answer>."
            ),
        )

    def student_refinement_prompt(obs, feedback: str) -> str:
        problem = obs["obs"]["problem"]
        if prompt_args.feedback_mode == "feedback":
            return _join_prompt_sections(
                render_task_block("countdown arithmetic"),
                f"Problem:\n{problem}",
                f"Feedback:\n{feedback}",
                "Revise your previous solution so it uses the feedback.",
                render_student_response_instruction("Put the final answer in <answer>...</answer>."),
            )
        if prompt_args.feedback_mode == "no_feedback":
            return _join_prompt_sections(
                render_task_block("countdown arithmetic"),
                f"Problem:\n{problem}",
                "Revise your previous expression.",
                render_student_response_instruction(
                    "Your final expression must appear exactly once in <answer>...</answer>."
                ),
            )
        raise ValueError(f"Unsupported feedback_mode: {prompt_args.feedback_mode}")

    def teacher_initial_prompt(obs, previous_attempt: str) -> str:
        problem = obs["obs"]["problem"]
        return _join_prompt_sections(
            render_task_block("countdown arithmetic"),
            f"I am trying to solve this question:\n{problem}",
            f"Here is my current attempt:\n{previous_attempt}",
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
            render_task_block("countdown arithmetic"),
            f"Here is my current attempt:\n{previous_attempt}",
            render_teacher_output_instruction(
                prompt_args,
                extra_rules=[
                    "Pinpoint the step in which I am making a mistake.",
                    "Provide the most informative piece of information for me to succeed on the next try, without telling me the final answer.",
                ],
            ),
        )

    return RefinementPromptSpec(
        system_prompt=COUNTDOWN_SYSTEM_PROMPT,
        student_initial_prompt=student_initial_prompt,
        student_refinement_prompt=student_refinement_prompt,
        teacher_initial_prompt=teacher_initial_prompt,
        teacher_refinement_prompt=teacher_refinement_prompt,
        teacher_system_prompt=DEFAULT_TEACHER_SYSTEM_PROMPT,
    )
