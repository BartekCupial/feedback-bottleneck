from feedback_bottleneck.config.args import RefinementPromptArgs
from feedback_bottleneck.llm.agent.refinement import (
    DEFAULT_TEACHER_SYSTEM_PROMPT,
    RefinementPromptSpec,
    _join_prompt_sections,
    render_ground_truth,
    render_student_response_instruction,
    render_teacher_output_instruction,
)

MATH_SYSTEM_PROMPT = "You are a mathematician solving competition-level math problems."


def get_instruction_prompt(task: str):
    del task
    return (
        "Solve the math problem by producing your current best answer. "
        "A correct answer gets reward +1. The episode ends on success or when the attempt limit is reached."
    )


def get_refinement_prompt_spec(
    task: str,
    *,
    prompt_args: RefinementPromptArgs,
) -> RefinementPromptSpec:
    del task

    final_answer_instruction = render_student_response_instruction(
        "End with exactly one final line in this format:\n\\boxed{your answer}"
    )

    def render_reference_context(obs) -> str:
        return render_ground_truth(
            prompt_args,
            solution_text=f"Here is the ground-truth solution:\n{obs['obs']['solution']}",
            answer_text=f"Here is the ground-truth final answer:\n{obs['obs']['answer']}",
        )

    def student_initial_prompt(obs) -> str:
        problem = obs["obs"]["problem"]
        return _join_prompt_sections(
            f"Problem:\n{problem}",
            "Solve the problem carefully.",
            final_answer_instruction,
        )

    def student_refinement_prompt(obs, feedback: str) -> str:
        del obs
        if prompt_args.feedback_mode == "feedback":
            return _join_prompt_sections(
                f"Feedback:\n{feedback}",
                "Revise your previous solution so it uses the feedback.",
                final_answer_instruction,
            )
        if prompt_args.feedback_mode == "no_feedback":
            return _join_prompt_sections(
                "Revise your previous solution and produce a corrected solution.",
                final_answer_instruction,
            )
        raise ValueError(f"Unsupported feedback_mode: {prompt_args.feedback_mode}")

    def teacher_initial_prompt(obs, previous_attempt: str) -> str:
        problem = obs["obs"]["problem"]
        return _join_prompt_sections(
            f"I am trying to solve this question:\n{problem}",
            f"Here is my current attempt:\n{previous_attempt}",
            render_reference_context(obs),
            render_teacher_output_instruction(
                prompt_args,
                extra_rules=[
                    "Pinpoint the step in which I am making a mistake.",
                    "Provide the most informative piece of information for me to succeed on the next try, without telling me the final answer.",
                ],
            ),
        )

    def teacher_refinement_prompt(obs, previous_attempt: str) -> str:
        del obs
        return _join_prompt_sections(
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
        system_prompt=MATH_SYSTEM_PROMPT,
        student_initial_prompt=student_initial_prompt,
        student_refinement_prompt=student_refinement_prompt,
        teacher_initial_prompt=teacher_initial_prompt,
        teacher_refinement_prompt=teacher_refinement_prompt,
        teacher_system_prompt=DEFAULT_TEACHER_SYSTEM_PROMPT,
    )
