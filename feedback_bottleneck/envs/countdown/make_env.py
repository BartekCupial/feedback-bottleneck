from typing import Optional

import gymnasium as gym

from feedback_bottleneck.envs.countdown.countdown import compute_score
from feedback_bottleneck.envs.env_wrapper import EnvWrapper


class IdentityLanguageActionSpace:
    def __init__(self, max_action_length: int):
        self.max_action_length = max_action_length
        self._values: list[str] = []

    def __contains__(self, action: str) -> bool:
        return isinstance(action, str)

    def map(self, action: str) -> str:
        text = "" if action is None else str(action)
        return text[: self.max_action_length].strip()


class CountdownEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_name, task, config, render_mode: Optional[str] = None, llm_judge=None):
        del env_name, task, render_mode, llm_judge
        super().__init__()
        self.config = config
        self.max_steps = int(config.countdown_args.max_turns)
        self.language_action_space = IdentityLanguageActionSpace(int(config.countdown_args.max_action_length))
        self.default_action = ""
        self.action_space = gym.spaces.Text(max_length=self.language_action_space.max_action_length)
        self.observation_space = gym.spaces.Dict({})
        self.actions = []

        self._episode_index = 0
        self._current = None
        self._attempts: list[dict] = []

    def get_wrapper_attr(self, name: str):
        return getattr(self, name)

    def get_text_action(self, action):
        return action

    def check_action_validity(self, candidate_action):
        return self.language_action_space.map(candidate_action), None

    def get_stats(self):
        return {"max_turns": self.max_steps, "num_attempts": len(self._attempts)}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        super().reset(seed=seed)
        self._attempts = []

        if seed is None:
            self._episode_index += 1
            seed = self._episode_index

        numbers, target = self._generate_countdown_instance()
        problem = (
            f"Using numbers {numbers}, create an expression that equals {target}. "
            "Use +, -, *, / and do not use any number more times than provided."
        )
        source = "countdown_generated"
        problem_id = f"generated-{seed}"

        self._current = {
            "problem": str(problem),
            "target": target,
            "numbers": numbers,
            "source": str(source),
            "problem_id": str(problem_id),
        }

        return self._build_observation(), {"correct": False, "source": self._current["source"]}

    def _generate_countdown_instance(self):
        # Generate a guaranteed-solvable target by composing operations over sampled numbers.
        numbers = [int(self.np_random.integers(1, 11)) for _ in range(4)]
        value = numbers[0]
        for n in numbers[1:]:
            ops = ["+", "-", "*"]
            if n != 0 and value % n == 0:
                ops.append("/")
            op = ops[int(self.np_random.integers(0, len(ops)))]

            if op == "+":
                value = value + n
            elif op == "-":
                value = value - n
            elif op == "*":
                value = value * n
            else:
                value = value // n

        return numbers, float(value)

    def step(self, action: str):
        if self._current is None:
            raise RuntimeError("Environment must be reset before stepping.")

        answer = self.language_action_space.map(action)
        reward = float(
            compute_score(
                solution_str=answer,
                ground_truth={"target": self._current["target"], "numbers": self._current["numbers"]},
            )
        )
        correct = reward == 1.0

        self._attempts.append(
            {
                "turn": len(self._attempts) + 1,
                "answer": answer,
                "correct": correct,
                "feedback": "Correct." if correct else "Incorrect.",
            }
        )

        terminated = correct
        truncated = (not terminated) and (len(self._attempts) >= self.max_steps)
        remaining = max(self.max_steps - len(self._attempts), 0)

        info = {
            "correct": correct,
            "problem_id": self._current["problem_id"],
            "source": self._current["source"],
            "attempts_used": len(self._attempts),
            "remaining_turns": remaining,
            "episode_extra_stats": {
                "correct": float(correct),
                "solved": float(correct),
                "attempts_used": len(self._attempts),
                "remaining_turns": remaining,
            },
        }
        if terminated:
            info["end_status"] = "success"
        elif truncated:
            info["end_status"] = "max_turns"

        return self._build_observation(), reward, terminated, truncated, info

    def _build_observation(self):
        attempts_text = "\n".join([f"Attempt {a['turn']}: {a['answer']}" for a in self._attempts])
        feedback_text = "\n".join([f"Attempt {a['turn']} verdict: {a['feedback']}" for a in self._attempts])
        if not attempts_text:
            attempts_text = "No previous attempts."
        if not feedback_text:
            feedback_text = "No judge feedback yet."

        remaining = self.max_steps - len(self._attempts)
        return {
            "obs": {
                "problem": self._current["problem"],
                "problem_id": self._current["problem_id"],
                "source": self._current["source"],
                "target": self._current["target"],
                "numbers": list(self._current["numbers"]),
                "attempts": list(self._attempts),
                "attempts_remaining": remaining,
                "num_attempts": len(self._attempts),
                "last_feedback": feedback_text,
            },
            "text": {
                "short_term_context": feedback_text,
                "long_term_context": (
                    f"Problem:\n{self._current['problem']}\n"
                    f"Attempts remaining: {remaining}\n"
                    f"Previous attempts:\n{attempts_text}"
                ),
            },
        }


def make_countdown_env(env_name, task, config, render_mode: Optional[str] = None, llm_judge=None):
    env = CountdownEnv(env_name, task, config, render_mode=render_mode, llm_judge=llm_judge)
    return EnvWrapper(env, env_name, task, args=config)
