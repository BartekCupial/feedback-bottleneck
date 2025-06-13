from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Dict, Optional

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.omni_code.instruction_prompt import get_refinement_prompt_spec
from feedback_bottleneck.envs.omni_code.tasks import parse_jsonish
from feedback_bottleneck.llm.agent.refinement import (
    HierarchicalRefinementAgent,
    HierarchicalRefinementFormatter,
    HierarchicalSeparateRefinementAgent,
    NaiveRefinementAgent,
    NaiveRefinementFormatter,
    _text_message,
    _trim_history_for_user_turn,
)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text


def _build_code_observation(step_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "obs": {
            "problem": step_data["obs_prompt"],
            "problem_id": step_data["obs_task_id"],
            "source": step_data["obs_source"],
            "difficulty": step_data["obs_difficulty"],
            "starter_code": step_data["obs_starter_code"],
            "reference_solution": step_data["obs_reference_solution"],
            "tests": parse_jsonish(step_data["obs_tests"]),
            "attempts": step_data["obs_attempts"],
            "latest_attempt": parse_jsonish(step_data.get("obs_latest_attempt")),
            "latest_failure": parse_jsonish(step_data.get("obs_latest_failure")),
            "attempts_remaining": step_data["obs_attempts_remaining"],
            "num_attempts": step_data["obs_num_attempts"],
            "last_feedback": step_data["obs_last_feedback"],
        },
        "text": {
            "short_term_context": step_data["short_term_context"],
            "long_term_context": step_data["long_term_context"],
        },
    }


class NaiveCodeFormatter(NaiveRefinementFormatter):
    def __init__(self, args: Args):
        prompt_args = replace(args.refinement_prompt, feedback_mode="no_feedback")
        super().__init__(
            args,
            prompt_spec=get_refinement_prompt_spec(
                args.task,
                prompt_args=prompt_args,
            ),
            step_observation_builder=_build_code_observation,
        )

    def append_user_turn(self, obs):
        if self.act_history:
            prompt_text = self.prompt_spec.student_refinement_prompt(obs, obs["obs"]["last_feedback"])
        else:
            prompt_text = self.prompt_spec.student_initial_prompt(obs)
        self.chat_history.append(_text_message("user", prompt_text))
        _trim_history_for_user_turn(self.chat_history, self.max_history)


class HierarchicalCodeFormatter(HierarchicalRefinementFormatter):
    def __init__(self, args: Args):
        super().__init__(
            args,
            prompt_spec=get_refinement_prompt_spec(
                args.task,
                prompt_args=args.refinement_prompt,
            ),
            step_observation_builder=_build_code_observation,
            normalize_action_for_change=extract_python_code,
        )


class NaiveCodeAgent(NaiveRefinementAgent):
    def _extract_action(self, solution: str) -> str:
        return extract_python_code(solution)


class HierarchicalCodeAgent(HierarchicalRefinementAgent):
    def _extract_action(self, solution: str) -> str:
        return extract_python_code(solution)


class HierarchicalSeparateCodeAgent(HierarchicalSeparateRefinementAgent):
    def _extract_action(self, solution: str) -> str:
        return extract_python_code(solution)
