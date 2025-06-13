from feedback_bottleneck.config.args import RefinementPromptArgs
from feedback_bottleneck.llm.agent.refinement import (
    DEFAULT_TEACHER_SYSTEM_PROMPT,
    RefinementPromptSpec,
    _join_prompt_sections,
    render_ground_truth,
    render_student_response_instruction,
    render_teacher_output_instruction,
)

SCIKNOWEVAL_SYSTEM_PROMPT = (
    "You are solving scientific knowledge and reasoning problems. "
    "Think carefully and return your final answer inside <answer>...</answer>."
)


def _format_metadata(obs) -> str:
    task_name = obs["obs"]["task_name"]
    domain = obs["obs"]["domain"]
    level = obs["obs"]["level"]
    subtask = obs["obs"]["subtask"]
    return "\n".join(
        [
            f"Domain: {domain}",
            f"Level: {level}",
            f"Task: {task_name}",
            f"Subtask: {subtask}",
        ]
    )


def get_instruction_prompt(task: str):
    if task == "default":
        task_name = "supported objective SciKnowEval tasks"
    else:
        task_name = f"the SciKnowEval task `{task}`"

    return (
        f"Solve questions from {task_name}. "
        "Provide your current best final answer. "
        "A correct answer gets reward +1 and the episode ends on success or when the attempt limit is reached."
    )


def get_refinement_prompt_spec(task: str, *, prompt_args: RefinementPromptArgs) -> RefinementPromptSpec:
    del task

    def student_initial_prompt(obs) -> str:
        metadata = _format_metadata(obs)
        problem = obs["obs"]["problem"]
        return _join_prompt_sections(
            metadata,
            f"Problem:\n{problem}",
            "Solve the problem carefully.",
            "For multiple-choice questions, you may answer with the option letter or the exact option text.",
            render_student_response_instruction("Put the final answer in <answer>...</answer>."),
        )

    def student_refinement_prompt(obs, feedback: str) -> str:
        metadata = _format_metadata(obs)
        if prompt_args.feedback_mode == "feedback":
            return _join_prompt_sections(
                metadata,
                f"Feedback:\n{feedback}",
                "Revise your previous solution so it uses the feedback.",
                "For multiple-choice questions, you may answer with the option letter or the exact option text.",
                render_student_response_instruction("Put the final answer in <answer>...</answer>."),
            )
        if prompt_args.feedback_mode == "no_feedback":
            return _join_prompt_sections(
                metadata,
                "Revise your previous solution so it uses the feedback.",
                "For multiple-choice questions, you may answer with the option letter or the exact option text.",
                render_student_response_instruction("Put the final answer in <answer>...</answer>."),
            )
        raise ValueError(f"Unsupported feedback_mode: {prompt_args.feedback_mode}")

    def teacher_initial_prompt(obs, previous_attempt: str) -> str:
        metadata = _format_metadata(obs)
        problem = obs["obs"]["problem"]
        reference_answer = obs["obs"]["answer"]
        return _join_prompt_sections(
            metadata,
            f"I am trying to solve this problem:\n{problem}",
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
        metadata = _format_metadata(obs)
        return _join_prompt_sections(
            metadata,
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
        system_prompt=SCIKNOWEVAL_SYSTEM_PROMPT,
        student_initial_prompt=student_initial_prompt,
        student_refinement_prompt=student_refinement_prompt,
        teacher_initial_prompt=teacher_initial_prompt,
        teacher_refinement_prompt=teacher_refinement_prompt,
        teacher_system_prompt=DEFAULT_TEACHER_SYSTEM_PROMPT,
    )
