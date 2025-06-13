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

BBEH_SYSTEM_PROMPT = (
    "You are solving hard reasoning questions. "
    "Think carefully and return your final answer inside <answer>...</answer>."
)


def get_instruction_prompt(task: str):
    task_name = "the full BBEH benchmark" if task == "default" else f"the BBEH task `{task}`"
    return (
        f"Solve questions from {task_name}. "
        "Provide your current best final answer. "
        "A correct answer gets reward +1 and the episode ends on success or when the attempt limit is reached."
    )


def get_refinement_prompt_spec(task: str, *, prompt_args: RefinementPromptArgs) -> RefinementPromptSpec:
    del task

    def student_initial_prompt(obs) -> str:
        task_name = obs["obs"]["task_name"]
        problem = obs["obs"]["problem"]
        return _join_prompt_sections(
            render_task_block(task_name),
            f"Problem:\n{problem}",
            "Solve the problem carefully.",
            render_student_response_instruction("Put the final answer in <answer>...</answer>."),
        )

    def student_refinement_prompt(obs, feedback: str) -> str:
        task_name = obs["obs"]["task_name"]
        if prompt_args.feedback_mode == "feedback":
            return _join_prompt_sections(
                render_task_block(task_name),
                f"Feedback:\n{feedback}",
                "Revise your previous solution so it uses the feedback.",
                render_student_response_instruction("Put the final answer in <answer>...</answer>."),
            )
        if prompt_args.feedback_mode == "no_feedback":
            return _join_prompt_sections(
                render_task_block(task_name),
                "Revise your previous solution and produce a corrected solution.",
                render_student_response_instruction("Put the final answer in <answer>...</answer>."),
            )
        raise ValueError(f"Unsupported feedback_mode: {prompt_args.feedback_mode}")

    def teacher_initial_prompt(obs, previous_attempt: str) -> str:
        task_name = obs["obs"]["task_name"]
        problem = obs["obs"]["problem"]
        reference_answer = obs["obs"]["answer"]
        return _join_prompt_sections(
            render_task_block(task_name),
            f"I am trying to solve this question:\n{problem}",
            f"Here is my current attempt:\n{previous_attempt}",
            render_ground_truth(
                prompt_args,
                answer_text=f"Here is the ground-truth reference answer:\n{reference_answer}",
            ),
            render_teacher_output_instruction(
                prompt_args,
                extra_rules=[
                    "Pinpoint the step in which I am making a mistake.",
                    "Provide the most informative piece of information for me to succeed on the next try, without telling me the final answer.",
                ],
            ),
        )

    def teacher_refinement_prompt(obs, previous_attempt: str) -> str:
        task_name = obs["obs"]["task_name"]
        return _join_prompt_sections(
            render_task_block(task_name),
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
        system_prompt=BBEH_SYSTEM_PROMPT,
        student_initial_prompt=student_initial_prompt,
        student_refinement_prompt=student_refinement_prompt,
        teacher_initial_prompt=teacher_initial_prompt,
        teacher_refinement_prompt=teacher_refinement_prompt,
        teacher_system_prompt=DEFAULT_TEACHER_SYSTEM_PROMPT,
    )
