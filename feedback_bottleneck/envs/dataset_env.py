from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Sequence

import gymnasium as gym

from feedback_bottleneck.envs.dataset_utils import ExampleT


class IdentityLanguageActionSpace:
    def __init__(self, max_action_length: int):
        self.max_action_length = max_action_length
        self._values: list[str] = []

    def __contains__(self, action: str) -> bool:
        return isinstance(action, str)

    def map(self, action: str) -> str:
        text = "" if action is None else str(action)
        return text[: self.max_action_length].strip()


class DatasetBackedTextEnv(gym.Env, ABC, Generic[ExampleT]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env_name: str,
        task: str,
        config: Any,
        examples: Sequence[ExampleT],
        max_steps: int,
        max_action_length: int,
    ):
        super().__init__()
        self.env_name = env_name
        self.task = task
        self.config = config
        self.max_steps = max_steps
        self.max_action_length = max_action_length
        self.language_action_space = IdentityLanguageActionSpace(self.max_action_length)
        self.default_action = ""
        self.action_space = gym.spaces.Text(max_length=self.max_action_length)
        self.observation_space = gym.spaces.Dict({})
        self.actions = []
        self.dataset = list(examples)
        if not self.dataset:
            raise ValueError(f"{env_name} requires at least one dataset example.")

        self._episode_index = 0
        self._current_example: Optional[ExampleT] = None
        self._attempts: list[dict[str, Any]] = []

    def get_wrapper_attr(self, name: str):
        return getattr(self, name)

    def get_text_action(self, action):
        return action

    def check_action_validity(self, candidate_action):
        return self.language_action_space.map(candidate_action), None

    def get_stats(self):
        return {
            "max_turns": self.max_steps,
            "num_attempts": len(self._attempts),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        super().reset(seed=seed)
        self._attempts = []

        if seed is None:
            idx = self._episode_index % len(self.dataset)
            self._episode_index += 1
        else:
            idx = int(seed) % len(self.dataset)

        self._current_example = self.dataset[idx]
        return self._build_observation(), self._build_reset_info()

    def step(self, action: str):
        example = self._require_current_example()
        answer = self.language_action_space.map(action)
        correct, feedback = self.evaluate_answer(answer, example)

        self._attempts.append(
            {
                "turn": len(self._attempts) + 1,
                "answer": answer,
                "correct": correct,
                "feedback": feedback,
            }
        )

        return self._build_transition(correct)

    def override_last_attempt(
        self,
        *,
        correct: bool,
        feedback: str,
    ):
        if not self._attempts:
            raise RuntimeError("Cannot override verification before any attempt was recorded.")

        self._attempts[-1]["correct"] = correct
        self._attempts[-1]["feedback"] = feedback
        return self._build_transition(correct)

    def _build_transition(self, correct: bool):
        reward = 1.0 if correct else 0.0
        terminated = bool(correct)
        truncated = (not terminated) and len(self._attempts) >= self.max_steps
        return (
            self._build_observation(),
            reward,
            terminated,
            truncated,
            self._build_info(
                correct=correct,
                terminated=terminated,
                truncated=truncated,
            ),
        )

    def _build_reset_info(self) -> dict[str, Any]:
        example = self._require_current_example()
        info = {
            "problem_id": example.problem_id,
            "source": example.source,
            "correct": False,
        }
        info.update(self.get_example_info(example))
        return info

    def _build_info(
        self,
        *,
        correct: bool,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        example = self._require_current_example()
        remaining_turns = max(self.max_steps - len(self._attempts), 0)
        info = {
            "correct": correct,
            "answer": example.answer,
            "problem_id": example.problem_id,
            "source": example.source,
            "attempts_used": len(self._attempts),
            "remaining_turns": remaining_turns,
            "episode_extra_stats": {
                "correct": float(correct),
                "solved": float(correct),
                "attempts_used": len(self._attempts),
                "remaining_turns": remaining_turns,
            },
        }
        info.update(self.get_example_info(example))

        if terminated:
            info["end_status"] = "success"
        elif truncated:
            info["end_status"] = "max_turns"

        return info

    def _format_attempt_history(self) -> tuple[str, str]:
        attempts_lines = []
        feedback_lines = []
        for attempt in self._attempts:
            attempts_lines.append(f"Attempt {attempt['turn']}: {attempt['answer']}")
            feedback_lines.append(f"Attempt {attempt['turn']} verdict: {attempt['feedback']}")

        attempts_text = "\n".join(attempts_lines) if attempts_lines else "No previous attempts."
        feedback_text = "\n".join(feedback_lines) if feedback_lines else "No judge feedback yet."
        return attempts_text, feedback_text

    def _require_current_example(self) -> ExampleT:
        if self._current_example is None:
            raise RuntimeError("Environment must be reset before stepping.")
        return self._current_example

    def get_example_info(self, example: ExampleT) -> dict[str, Any]:
        del example
        return {}

    @abstractmethod
    def evaluate_answer(self, answer: str, example: ExampleT) -> tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def _build_observation(self) -> dict[str, Any]:
        raise NotImplementedError
