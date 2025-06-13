import re
from typing import Any, Dict

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.arc_agi.instruction_prompt import get_refinement_prompt_spec as get_arc_agi_prompt_spec
from feedback_bottleneck.envs.omni_code.tasks import parse_jsonish
from feedback_bottleneck.llm.agent.refinement import (
    HierarchicalRefinementAgent,
    HierarchicalRefinementFormatter,
    HierarchicalSeparateRefinementAgent,
    NaiveRefinementAgent,
    NaiveRefinementFormatter,
)


def _extract_tagged_text(solution_text: str | None, tag: str) -> str:
    if solution_text is None:
        return ""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", str(solution_text), re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return str(solution_text).strip()


def _build_arc_agi_observation(step_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "obs": {
            "problem": step_data["obs_problem"],
            "problem_id": step_data["obs_problem_id"],
            "source": step_data["obs_source"],
            "split": step_data["obs_split"],
            "train": parse_jsonish(step_data["obs_train"]),
            "test_inputs": parse_jsonish(step_data["obs_test_inputs"]),
            "answer": step_data["obs_answer"],
            "attempts": parse_jsonish(step_data["obs_attempts"]),
            "latest_attempt": parse_jsonish(step_data.get("obs_latest_attempt")),
            "attempts_remaining": step_data["obs_attempts_remaining"],
            "num_attempts": step_data["obs_num_attempts"],
            "last_feedback": step_data["obs_last_feedback"],
        },
        "text": {
            "short_term_context": step_data["short_term_context"],
            "long_term_context": step_data["long_term_context"],
        },
    }


class NaiveArcAgiFormatter(NaiveRefinementFormatter):
    def __init__(self, args: Args):
        super().__init__(
            args,
            prompt_spec=get_arc_agi_prompt_spec(args.task, prompt_args=args.refinement_prompt),
            step_observation_builder=_build_arc_agi_observation,
        )


class HierarchicalArcAgiFormatter(HierarchicalRefinementFormatter):
    def __init__(self, args: Args):
        super().__init__(
            args,
            prompt_spec=get_arc_agi_prompt_spec(args.task, prompt_args=args.refinement_prompt),
            step_observation_builder=_build_arc_agi_observation,
            normalize_action_for_change=lambda text: _extract_tagged_text(text, "answer"),
        )

    def append_student_prompt(self, obs: Dict[str, Any], prev_solution: str | None, feedback: str | None):
        # ARC tasks can be large; each turn already contains the full task JSON and structured feedback.
        # Carrying prior full prompts in chat history quickly exceeds model context on multi-turn runs.
        self.student_chat_history.clear()
        super().append_student_prompt(obs, prev_solution, feedback)

    def append_teacher_prompt(self, obs: Dict[str, Any], prev_solution: str):
        self.teacher_chat_history.clear()
        super().append_teacher_prompt(obs, prev_solution)


class NaiveArcAgiAgent(NaiveRefinementAgent):
    def __init__(self, llm_actor, prompt_formatter, args: Args):
        super().__init__(llm_actor, prompt_formatter, args, answer_tag="answer")


class HierarchicalArcAgiAgent(HierarchicalRefinementAgent):
    def __init__(self, llm_actor, prompt_formatter, args: Args):
        super().__init__(llm_actor, prompt_formatter, args, answer_tag="answer")


class HierarchicalSeparateArcAgiAgent(HierarchicalSeparateRefinementAgent):
    def __init__(self, llm_judge, llm_actor, prompt_formatter, args: Args):
        super().__init__(llm_judge, llm_actor, prompt_formatter, args, answer_tag="answer")
