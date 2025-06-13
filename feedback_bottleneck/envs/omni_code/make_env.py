from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.dataset.loading import load_single_split_dataset
from feedback_bottleneck.envs.env_wrapper import EnvWrapper

from .sandbox import SubprocessCodeSandbox
from .tasks import (
    CodeTask,
    CodeTestSpec,
    filter_dataset_by_problem_split,
    normalize_task_record,
    record_matches_code_filters,
    resolve_execution_timeouts,
)

logger = logging.getLogger(__name__)


def _extract_latest_failure(result) -> dict[str, Any] | None:
    first_failure = next((case for case in result.cases if not case.passed), None)
    if first_failure is None:
        return None
    return {
        "index": first_failure.index,
        "error_type": first_failure.error_type,
        "input_preview": first_failure.input_preview,
        "expected_preview": first_failure.expected_preview,
        "actual_preview": first_failure.actual_preview,
        "stdout": first_failure.stdout,
        "stderr": first_failure.stderr,
    }


@dataclass
class IdentityLanguageActionSpace:
    max_action_length: int
    _values: list[str] = field(default_factory=list)

    def __contains__(self, action: str) -> bool:
        return isinstance(action, str)

    def map(self, action: str) -> str:
        action = "" if action is None else str(action)
        return action[: self.max_action_length].strip()


def load_code_dataset(data_path: str, dataset_config: Optional[str], dataset_split: str):
    dataset = load_single_split_dataset(data_path, name=dataset_config, split=dataset_split)
    logging.info("Loaded raw code dataset with %s rows", len(dataset))
    return dataset


def _row_to_task(row: dict[str, Any]) -> CodeTask:
    tests = row["tests"]
    return CodeTask(
        task_id=row["task_id"],
        prompt=row["prompt"],
        starter_code=row.get("starter_code", ""),
        reference_solution=row["reference_solution"],
        tests=CodeTestSpec(
            kind=tests["kind"],
            inputs=list(tests.get("inputs", [])),
            outputs=list(tests.get("outputs", [])),
            fn_name=tests.get("fn_name"),
            asserts=list(tests.get("asserts", [])),
        ),
        source=row.get("source", ""),
        difficulty=row.get("difficulty", ""),
        metadata=row.get("metadata", {}),
    )


class CodeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_name: str,
        task: str,
        config: Args,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.env_name = env_name
        self.task = task
        self.config = config
        self.render_mode = render_mode
        self.max_steps = int(config.code_args.max_turns)
        self.max_action_length = int(config.code_args.max_action_length)
        self.language_action_space = IdentityLanguageActionSpace(self.max_action_length)
        self.default_action = ""
        self.action_space = gym.spaces.Text(max_length=self.max_action_length)
        self.observation_space = gym.spaces.Dict({})
        self.actions = []
        self.sandbox = SubprocessCodeSandbox(default_timeout_s=float(config.code_args.sandbox_case_timeout_s))

        raw_dataset = load_code_dataset(
            config.code_args.dataset_path,
            config.code_args.dataset_config,
            config.code_args.dataset_split,
        )

        normalized_rows = []
        source_name = config.code_args.source_name or config.code_args.dataset_path
        for idx, row in enumerate(raw_dataset):
            record = dict(row)
            if not record_matches_code_filters(
                record,
                min_rating=config.code_args.min_rating,
                max_rating=config.code_args.max_rating,
                require_official_tests_complete=bool(config.code_args.require_official_tests_complete),
                require_stdio=bool(config.code_args.require_stdio),
                require_no_generated_checker=bool(config.code_args.require_no_generated_checker),
                required_tags=list(config.code_args.required_tags),
            ):
                continue
            task_row = normalize_task_record(
                record,
                idx,
                source_name=source_name,
                max_cases=config.code_args.max_cases,
                require_reference_solution=False,
            )
            if task_row is not None:
                normalized_rows.append(task_row.to_dict())

        if not normalized_rows:
            raise ValueError("No valid code tasks found after normalization.")

        self.dataset = filter_dataset_by_problem_split(
            normalized_rows,
            split_name=config.code_args.problem_split,
            split_seed=config.code_args.problem_split_seed,
            test_size=config.code_args.problem_split_test_size,
        )

        self._episode_index = 0
        self._current_task: Optional[CodeTask] = None
        self._attempts: list[dict[str, Any]] = []

    def get_wrapper_attr(self, name: str):
        return getattr(self, name)

    def get_text_action(self, action):
        return action

    def check_action_validity(self, candidate_action):
        return self.language_action_space.map(candidate_action), None

    def get_stats(self):
        best_pass_rate = max((attempt["pass_rate"] for attempt in self._attempts), default=0.0)
        best_success = max((attempt["solved"] for attempt in self._attempts), default=False)
        pass_at_2 = any(attempt["solved"] for attempt in self._attempts[:2])
        stats = {
            "max_turns": self.max_steps,
            "num_attempts": len(self._attempts),
            "pass_at_2": float(pass_at_2),
            "best_success": float(best_success),
            "best_pass_rate": best_pass_rate,
        }
        if self._current_task is not None:
            stats["difficulty"] = self._current_task.difficulty
        return stats

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._attempts = []

        if seed is None:
            idx = self._episode_index % len(self.dataset)
            self._episode_index += 1
        else:
            idx = int(seed) % len(self.dataset)

        self._current_task = _row_to_task(dict(self.dataset[idx]))
        obs = self._build_observation()
        info = {
            "task_id": self._current_task.task_id,
            "source": self._current_task.source,
            "difficulty": self._current_task.difficulty,
            "correct": False,
        }
        return obs, info

    def step(self, action: str):
        if self._current_task is None:
            raise RuntimeError("Environment must be reset before stepping.")

        candidate_code = self.language_action_space.map(action)
        case_timeout_s, task_timeout_s = resolve_execution_timeouts(
            self._current_task,
            default_case_timeout_s=float(self.config.code_args.sandbox_case_timeout_s),
            default_task_timeout_s=self.config.code_args.sandbox_task_timeout_s,
            use_dataset_time_limit=bool(self.config.code_args.sandbox_use_dataset_time_limit),
            time_limit_multiplier=float(self.config.code_args.sandbox_time_limit_multiplier),
            min_case_timeout_s=self.config.code_args.sandbox_min_case_timeout_s,
            max_case_timeout_s=self.config.code_args.sandbox_max_case_timeout_s,
            task_timeout_case_budget_multiplier=float(
                self.config.code_args.sandbox_task_timeout_case_budget_multiplier
            ),
        )
        result = self.sandbox.run(
            self._current_task,
            candidate_code,
            timeout_s=case_timeout_s,
            task_timeout_s=task_timeout_s,
        )
        solved = result.passed
        reward = result.pass_rate

        attempt_no = len(self._attempts) + 1
        terminated = bool(solved)
        truncated = (not terminated) and attempt_no >= self.max_steps
        feedback = result.feedback(max_chars=int(self.config.code_args.feedback_max_chars))

        self._attempts.append(
            {
                "turn": attempt_no,
                "code": candidate_code,
                "solved": solved,
                "pass_rate": result.pass_rate,
                "num_passed": result.num_passed,
                "num_total": result.num_total,
                "feedback": feedback,
                "timeout": result.timeout,
                "case_timeout_s": case_timeout_s,
                "task_timeout_s": task_timeout_s,
                "latest_failure": _extract_latest_failure(result),
            }
        )

        obs = self._build_observation()
        best_pass_rate = max(attempt["pass_rate"] for attempt in self._attempts)
        info = {
            "correct": solved,
            "task_id": self._current_task.task_id,
            "source": self._current_task.source,
            "difficulty": self._current_task.difficulty,
            "attempts_used": len(self._attempts),
            "remaining_turns": max(self.max_steps - len(self._attempts), 0),
            "pass_rate": result.pass_rate,
            "tests_passed": result.num_passed,
            "tests_total": result.num_total,
            "episode_extra_stats": {
                "solved": float(solved),
                "correct": float(solved),
                "attempts_used": len(self._attempts),
                "remaining_turns": max(self.max_steps - len(self._attempts), 0),
                "pass_rate": result.pass_rate,
                "best_pass_rate": best_pass_rate,
                "tests_passed": result.num_passed,
                "tests_total": result.num_total,
            },
        }
        if terminated:
            info["end_status"] = "success"
        elif truncated:
            info["end_status"] = "max_turns"

        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> dict[str, Any]:
        task = self._current_task
        attempts_lines = []
        for attempt in self._attempts:
            attempts_lines.append(
                f"Attempt {attempt['turn']}: pass_rate={attempt['pass_rate']:.2f} "
                f"({attempt['num_passed']}/{attempt['num_total']})"
            )
        attempts_text = "\n".join(attempts_lines) if attempts_lines else "No previous attempts."
        latest_attempt = self._attempts[-1] if self._attempts else None
        last_feedback = latest_attempt["feedback"] if latest_attempt else "No execution feedback yet."
        latest_failure = latest_attempt["latest_failure"] if latest_attempt else None
        remaining_turns = self.max_steps - len(self._attempts)

        long_term_lines = [f"Problem:\n{task.prompt}"]
        if task.starter_code:
            long_term_lines.append(f"Starter code:\n{task.starter_code}")
        long_term_lines.extend(
            [
                f"Attempts remaining: {remaining_turns}",
                f"Attempt summary:\n{attempts_text}",
            ]
        )

        return {
            "obs": {
                "problem": task.prompt,
                "problem_id": task.task_id,
                "prompt": task.prompt,
                "task_id": task.task_id,
                "source": task.source,
                "difficulty": task.difficulty,
                "starter_code": task.starter_code,
                "reference_solution": task.reference_solution,
                "tests": task.tests.to_dict(),
                "attempts": list(self._attempts),
                "latest_attempt": dict(latest_attempt) if latest_attempt else None,
                "latest_failure": dict(latest_failure) if latest_failure else None,
                "attempts_remaining": remaining_turns,
                "num_attempts": len(self._attempts),
                "last_feedback": last_feedback,
            },
            "text": {
                "short_term_context": last_feedback,
                "long_term_context": "\n\n".join(long_term_lines),
            },
        }


def make_code_env(env_name, task, config, render_mode: Optional[str] = None):
    env = CodeEnv(
        env_name=env_name,
        task=task,
        config=config,
        render_mode=render_mode,
    )
    return EnvWrapper(env, env_name, task, args=config)
