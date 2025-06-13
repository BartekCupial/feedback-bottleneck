from feedback_bottleneck.config.args import RefinementPromptArgs
from feedback_bottleneck.llm.agent.refinement import (
    DEFAULT_TEACHER_SYSTEM_PROMPT,
    RefinementPromptSpec,
    _join_prompt_sections,
    render_ground_truth,
    render_task_block,
    render_teacher_output_instruction,
)

ARC_AGI_SYSTEM_PROMPT = (
    "You solve ARC-AGI grid reasoning tasks. Infer the transformation from the training pairs "
    "and return only the predicted output grid or grids for the test inputs."
)

ARC_AGI_ANSWER_INSTRUCTION = (
    "Return exactly one <answer>...</answer> block and no other text. "
    "Inside the tag, return JSON only. Use {\"outputs\": [[[...]], ...]} for multiple test inputs."
)


def get_instruction_prompt(task: str = "default") -> str:
    del task
    return (
        "Solve ARC-AGI tasks. Each task gives training input/output grid pairs and test input grids. "
        "Return predicted test output grids as JSON. The episode allows two attempts."
    )


def _render_task(obs) -> str:
    return _join_prompt_sections(
        f"Task id: {obs['obs']['problem_id']}",
        "Task JSON without test outputs:",
        obs["obs"]["problem"],
    )


def _render_reference_context(obs, prompt_args: RefinementPromptArgs) -> str:
    return render_ground_truth(
        prompt_args,
        answer_text=f"Hidden target output grids for diagnosis only:\n{obs['obs']['answer']}",
        none_message="You do not have access to the hidden target output grids.",
    )


def get_refinement_prompt_spec(
    task: str,
    *,
    prompt_args: RefinementPromptArgs,
) -> RefinementPromptSpec:
    del task

    def student_initial_prompt(obs) -> str:
        return _join_prompt_sections(
            render_task_block("ARC-AGI"),
            _render_task(obs),
            "Infer the rule from the train pairs and solve every test input.",
            ARC_AGI_ANSWER_INSTRUCTION,
        )

    def student_refinement_prompt(obs, feedback: str) -> str:
        if prompt_args.feedback_mode == "feedback":
            feedback_block = f"Feedback:\n{feedback}"
            action = "Produce a corrected ARC answer from the beginning using this feedback."
        elif prompt_args.feedback_mode == "no_feedback":
            feedback_block = f"Previous attempt feedback:\n{obs['obs']['last_feedback']}"
            action = "Produce a corrected ARC answer from the beginning."
        else:
            raise ValueError(f"Unsupported feedback_mode: {prompt_args.feedback_mode}")

        return _join_prompt_sections(
            render_task_block("ARC-AGI"),
            _render_task(obs),
            feedback_block,
            action,
            ARC_AGI_ANSWER_INSTRUCTION,
        )

    def teacher_initial_prompt(obs, previous_attempt: str) -> str:
        return _join_prompt_sections(
            render_task_block("ARC-AGI"),
            _render_task(obs),
            f"Here is my current attempt:\n{previous_attempt}",
            f"Execution-side feedback:\n{obs['obs']['last_feedback']}",
            _render_reference_context(obs, prompt_args),
            render_teacher_output_instruction(
                prompt_args,
                extra_rules=[
                    "Do not copy or reveal hidden target output grids, code, or final answers.",
                    "Give only diagnostic feedback that helps re-check the transformation.",
                ],
            ),
        )

    def teacher_refinement_prompt(obs, previous_attempt: str) -> str:
        return _join_prompt_sections(
            render_task_block("ARC-AGI"),
            _render_task(obs),
            f"Here is my current attempt:\n{previous_attempt}",
            f"Execution-side feedback:\n{obs['obs']['last_feedback']}",
            _render_reference_context(obs, prompt_args),
            render_teacher_output_instruction(
                prompt_args,
                extra_rules=[
                    "Do not copy or reveal hidden target output grids, code, or final answers.",
                    "Give only diagnostic feedback that helps re-check the transformation.",
                ],
            ),
        )

    return RefinementPromptSpec(
        system_prompt=ARC_AGI_SYSTEM_PROMPT,
        student_initial_prompt=student_initial_prompt,
        student_refinement_prompt=student_refinement_prompt,
        teacher_initial_prompt=teacher_initial_prompt,
        teacher_refinement_prompt=teacher_refinement_prompt,
        teacher_system_prompt=DEFAULT_TEACHER_SYSTEM_PROMPT,
    )
