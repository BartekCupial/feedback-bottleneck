from typing import Any, Dict

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.omni_math.instruction_prompt import get_refinement_prompt_spec
from feedback_bottleneck.llm.agent.refinement import (
    HierarchicalRefinementAgent,
    HierarchicalRefinementFormatter,
    HierarchicalSeparateRefinementAgent,
    NaiveRefinementAgent,
    NaiveRefinementFormatter,
)


def extract_last_boxed(text: str | None) -> str | None:
    if text is None:
        return None
    text = str(text)
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None
    return text[left_brace_idx + 1 : right_brace_idx].strip()


def _extract_answer_text(solution_text: str | None) -> str:
    boxed = extract_last_boxed(solution_text)
    if boxed is not None:
        return boxed
    return (solution_text or "").strip()


def _build_math_observation(step_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "obs": {
            "problem": step_data["obs_problem"],
            "problem_id": step_data["obs_problem_id"],
            "source": step_data["obs_source"],
            "solution": step_data["obs_solution"],
            "answer": step_data["obs_answer"],
            "attempts": step_data["obs_attempts"],
            "attempts_remaining": step_data["obs_attempts_remaining"],
            "num_attempts": step_data["obs_num_attempts"],
            "last_feedback": step_data["obs_last_feedback"],
        },
        "text": {
            "short_term_context": step_data["short_term_context"],
            "long_term_context": step_data["long_term_context"],
        },
    }


class NaiveMathFormatter(NaiveRefinementFormatter):
    def __init__(self, args: Args):
        super().__init__(
            args,
            prompt_spec=get_refinement_prompt_spec(
                args.task,
                prompt_args=args.refinement_prompt,
            ),
            step_observation_builder=_build_math_observation,
        )


class HierarchicalMathFormatter(HierarchicalRefinementFormatter):
    def __init__(self, args: Args):
        super().__init__(
            args,
            prompt_spec=get_refinement_prompt_spec(
                args.task,
                prompt_args=args.refinement_prompt,
            ),
            step_observation_builder=_build_math_observation,
            normalize_action_for_change=_extract_answer_text,
        )


class NaiveMathAgent(NaiveRefinementAgent):
    pass


class HierarchicalMathAgent(HierarchicalRefinementAgent):
    pass


class HierarchicalSeparateMathAgent(HierarchicalSeparateRefinementAgent):
    pass
