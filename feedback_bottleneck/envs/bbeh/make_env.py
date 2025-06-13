import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.bbeh.evaluate import evaluate_correctness
from feedback_bottleneck.envs.dataset_env import DatasetBackedTextEnv
from feedback_bottleneck.envs.dataset_utils import DatasetExample, load_dataset_examples
from feedback_bottleneck.envs.env_wrapper import EnvWrapper

logger = logging.getLogger(__name__)


@dataclass
class BBehExample(DatasetExample):
    task_name: str = ""


def _make_problem_id(task_name: str, problem: str, answer: str) -> str:
    digest = hashlib.sha1(f"{task_name}\n{problem}\n{answer}".encode("utf-8")).hexdigest()[:16]
    return f"{task_name}:{digest}"


def _task_aliases(task_name: str) -> set[str]:
    base = str(task_name).strip()
    normalized_space = " ".join(base.replace("_", " ").split()).lower()
    normalized_underscore = normalized_space.replace(" ", "_")
    return {
        base.lower(),
        normalized_space,
        normalized_underscore,
        f"bbeh_{normalized_underscore}",
    }


def _normalize_record(record: dict, idx: int) -> BBehExample:
    del idx
    problem = record.get("input") or record.get("question") or record.get("problem")
    answer = record.get("target") or record.get("answer")
    task_name = record.get("task_name") or record.get("task")
    if problem is None or answer is None or task_name is None:
        raise ValueError("Unsupported BBEH row: expected `task`, `input`, and `target` fields.")

    problem = str(problem).strip()
    answer = str(answer).strip()
    task_name = str(task_name).strip()
    source = str(record.get("source") or "bbeh_hf")

    return BBehExample(
        problem=problem,
        answer=answer,
        task_name=task_name,
        source=source,
        problem_id=_make_problem_id(task_name, problem, answer),
        raw_record=dict(record),
    )


def _filter_bbeh_dataset(dataset, task: str):
    if task == "mini":
        if "mini" not in dataset.column_names:
            raise ValueError("BBEH dataset does not expose a `mini` column needed for task='mini'.")
        dataset = dataset.filter(lambda row: int(row["mini"]) == 1)
        logger.info("Filtering subtask %s dataset with %s samples", task, len(dataset))
        return dataset

    if task != "default":
        if "task" not in dataset.column_names:
            raise ValueError("BBEH dataset does not expose a `task` column needed for task filtering.")
        aliases = _task_aliases(task)
        dataset = dataset.filter(lambda row: str(row["task"]).strip().lower() in aliases)
        logger.info("Filtering subtask %s dataset with %s samples", task, len(dataset))
        return dataset

    return dataset


def load_bbeh_examples(config: Args, task: str) -> list[BBehExample]:
    return load_dataset_examples(
        config.bbeh_args.dataset_path,
        config.bbeh_args.dataset_config,
        config.bbeh_args.dataset_split,
        normalize_record=_normalize_record,
        problem_split=config.bbeh_args.problem_split,
        problem_split_seed=config.bbeh_args.problem_split_seed,
        problem_split_test_size=config.bbeh_args.problem_split_test_size,
        split_label="bbeh",
        dataset_filter=lambda dataset: _filter_bbeh_dataset(dataset, task),
        empty_error_message=f"No BBEH examples found in {config.bbeh_args.dataset_path} for task '{task}'",
    )


class BBehEnv(DatasetBackedTextEnv[BBehExample]):
    def __init__(
        self,
        env_name: str,
        task: str,
        config: Args,
        render_mode: Optional[str] = None,
        llm_judge=None,
    ):
        del render_mode, llm_judge
        super().__init__(
            env_name=env_name,
            task=task,
            config=config,
            examples=load_bbeh_examples(config, task),
            max_steps=int(config.bbeh_args.max_turns),
            max_action_length=int(config.bbeh_args.max_action_length),
        )

    def get_example_info(self, example: BBehExample) -> dict[str, Any]:
        return {"task_name": example.task_name}

    def evaluate_answer(self, answer: str, example: BBehExample) -> tuple[bool, str]:
        correct = evaluate_correctness(answer, example.answer)
        return correct, "Correct." if correct else "Incorrect."

    def _build_observation(self) -> dict[str, Any]:
        example = self._require_current_example()
        attempts_text, verdict_text = self._format_attempt_history()
        remaining_turns = self.max_steps - len(self._attempts)

        return {
            "obs": {
                "problem": example.problem,
                "problem_id": example.problem_id,
                "source": example.source,
                "task_name": example.task_name,
                "answer": example.answer,
                "attempts": list(self._attempts),
                "attempts_remaining": remaining_turns,
                "num_attempts": len(self._attempts),
                "last_feedback": verdict_text,
            },
            "text": {
                "short_term_context": verdict_text,
                "long_term_context": (
                    f"Task: {example.task_name}\n"
                    f"Question:\n{example.problem}\n"
                    f"Attempts remaining: {remaining_turns}\n"
                    f"Previous attempts:\n{attempts_text}"
                ),
            },
        }


def make_bbeh_env(env_name, task, config, render_mode: Optional[str] = None, llm_judge=None):
    env = BBehEnv(env_name, task, config, render_mode=render_mode, llm_judge=llm_judge)
    return EnvWrapper(env, env_name, task, args=config)
