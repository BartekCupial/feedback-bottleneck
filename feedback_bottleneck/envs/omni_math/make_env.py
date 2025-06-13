from dataclasses import dataclass
from typing import Any, Optional

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.dataset_env import DatasetBackedTextEnv
from feedback_bottleneck.envs.dataset_utils import DatasetExample, load_dataset_examples
from feedback_bottleneck.envs.env_wrapper import EnvWrapper
from feedback_bottleneck.envs.omni_math.compute_score import compute_score
from feedback_bottleneck.envs.omni_math.llm_verify import MathLLMVerifier
from feedback_bottleneck.envs.omni_math.problem_split import make_problem_id


@dataclass
class MathExample(DatasetExample):
    solution: str = ""
    difficulty: Any = ""


def _normalize_record(record: dict, idx: int) -> MathExample:
    del idx
    problem = record["problem"]
    answer = record["answer"]
    solution = record["solution"]
    source = record["source"]
    difficulty = record["difficulty"]

    return MathExample(
        problem=str(problem).strip(),
        answer=str(answer).strip(),
        solution=str(solution).strip(),
        source=str(source).strip(),
        problem_id=make_problem_id(problem),
        raw_record=dict(record),
        difficulty=difficulty,
    )


def load_math_examples(config: Args) -> list[MathExample]:
    return load_dataset_examples(
        config.math_args.dataset_path,
        config.math_args.dataset_config,
        config.math_args.dataset_split,
        normalize_record=_normalize_record,
        problem_split=config.math_args.problem_split,
        problem_split_seed=config.math_args.problem_split_seed,
        problem_split_test_size=config.math_args.problem_split_test_size,
        split_label="math",
    )


class MathEnv(DatasetBackedTextEnv[MathExample]):
    def __init__(
        self,
        env_name: str,
        task: str,
        config: Args,
        render_mode: Optional[str] = None,
        llm_judge=None,
    ):
        super().__init__(
            env_name=env_name,
            task=task,
            config=config,
            examples=load_math_examples(config),
            max_steps=int(config.math_args.max_turns),
            max_action_length=int(config.math_args.max_action_length),
        )
        self.math_verifier = None

        if config.math_args.verification_mode == "llm":
            if llm_judge is None:
                raise ValueError("math_args.verification_mode='llm' requires llm_judge to be configured.")
            self.math_verifier = MathLLMVerifier(args=config, llm_judge=llm_judge)

    def get_stats(self):
        best_success = max((attempt["correct"] for attempt in self._attempts), default=False)
        pass_at_2 = any(attempt["correct"] for attempt in self._attempts[:2])
        stats = {
            "max_turns": self.max_steps,
            "num_attempts": len(self._attempts),
            "pass_at_2": float(pass_at_2),
            "best_success": float(best_success),
        }
        if self._current_example is not None:
            stats["difficulty"] = self._current_example.difficulty
        return stats

    def evaluate_answer(self, answer: str, example: MathExample) -> tuple[bool, str]:
        correct = compute_score(answer, example.answer) == 1.0
        return correct, "Correct." if correct else "Incorrect."

    async def apply_post_step_verification(
        self,
        action: str,
        obs,
        reward,
        terminated,
        truncated,
        info,
    ):
        if self.math_verifier is None:
            return obs, reward, terminated, truncated, info

        verification = await self.math_verifier.verify(
            observation=obs["obs"],
            candidate_answer=action,
        )

        return self.override_last_attempt_verification(
            correct=verification.correct,
            feedback=verification.feedback,
        )

    def override_last_attempt_verification(
        self,
        *,
        correct: bool,
        feedback: str,
    ):
        return self.override_last_attempt(correct=correct, feedback=feedback)

    def _build_observation(self) -> dict[str, Any]:
        example = self._require_current_example()
        attempts_text, judge_text = self._format_attempt_history()
        remaining_turns = self.max_steps - len(self._attempts)

        return {
            "obs": {
                "problem": example.problem,
                "problem_id": example.problem_id,
                "source": example.source,
                "solution": example.solution,
                "answer": example.answer,
                "attempts": list(self._attempts),
                "attempts_remaining": remaining_turns,
                "num_attempts": len(self._attempts),
                "last_feedback": judge_text,
            },
            "text": {
                "short_term_context": judge_text,
                "long_term_context": "\n".join(
                    [
                        f"Problem:\n{example.problem}",
                        f"Attempts remaining: {remaining_turns}",
                        f"Previous attempts:\n{attempts_text}",
                    ]
                ),
            },
        }


def make_math_env(env_name, task, config, render_mode: Optional[str] = None, llm_judge=None):
    env = MathEnv(
        env_name=env_name,
        task=task,
        config=config,
        render_mode=render_mode,
        llm_judge=llm_judge,
    )
    return EnvWrapper(env, env_name, task, args=config)
