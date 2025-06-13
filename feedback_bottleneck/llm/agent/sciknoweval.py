import re
from typing import Any, Dict

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.sciknoweval.instruction_prompt import get_refinement_prompt_spec
from feedback_bottleneck.llm.agent.refinement import (
    HierarchicalRefinementAgent,
    HierarchicalRefinementFormatter,
    HierarchicalSeparateRefinementAgent,
    NaiveRefinementAgent,
    NaiveRefinementFormatter,
)


def _extract_answer_text(solution_text: str | None) -> str:
    if solution_text is None:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", str(solution_text), re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return str(solution_text).strip()


def _build_sciknoweval_observation(step_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "obs": {
            "problem": step_data["obs_problem"],
            "problem_id": step_data["obs_problem_id"],
            "source": step_data["obs_source"],
            "task_name": step_data["obs_task_name"],
            "domain": step_data["obs_domain"],
            "level": step_data["obs_level"],
            "subtask": step_data["obs_subtask"],
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


class NaiveSciKnowEvalFormatter(NaiveRefinementFormatter):
    def __init__(self, args: Args):
        super().__init__(
            args,
            prompt_spec=get_refinement_prompt_spec(args.task, prompt_args=args.refinement_prompt),
            step_observation_builder=_build_sciknoweval_observation,
        )


class HierarchicalSciKnowEvalFormatter(HierarchicalRefinementFormatter):
    def __init__(self, args: Args):
        super().__init__(
            args,
            prompt_spec=get_refinement_prompt_spec(args.task, prompt_args=args.refinement_prompt),
            step_observation_builder=_build_sciknoweval_observation,
            normalize_action_for_change=_extract_answer_text,
        )


class NaiveSciKnowEvalAgent(NaiveRefinementAgent):
    def __init__(self, llm_actor, prompt_formatter, args: Args):
        super().__init__(llm_actor, prompt_formatter, args, answer_tag="answer")


class HierarchicalSciKnowEvalAgent(HierarchicalRefinementAgent):
    def __init__(self, llm_actor, prompt_formatter, args: Args):
        super().__init__(llm_actor, prompt_formatter, args, answer_tag="answer")


class HierarchicalSeparateSciKnowEvalAgent(HierarchicalSeparateRefinementAgent):
    def __init__(self, llm_judge, llm_actor, prompt_formatter, args: Args):
        super().__init__(llm_judge, llm_actor, prompt_formatter, args, answer_tag="answer")
