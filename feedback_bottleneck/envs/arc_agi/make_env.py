from __future__ import annotations

import json
import logging
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional

import gymnasium as gym

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.env_wrapper import EnvWrapper

logger = logging.getLogger(__name__)

ARC_AGI2_ZIP_URL = "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip"
ARC_AGI1_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"


@dataclass(frozen=True)
class ArcPair:
    input: list[list[int]]
    output: list[list[int]]


@dataclass(frozen=True)
class ArcAgiTask:
    task_id: str
    split: str
    source_path: str
    train: list[ArcPair]
    test: list[ArcPair]


@dataclass
class IdentityLanguageActionSpace:
    max_action_length: int
    _values: list[str] = field(default_factory=list)

    def __contains__(self, action: str) -> bool:
        return isinstance(action, str)

    def map(self, action: str) -> str:
        action = "" if action is None else str(action)
        return action[: self.max_action_length].strip()


def _validate_grid(value: Any) -> tuple[Optional[list[list[int]]], Optional[str]]:
    if not isinstance(value, list) or not value:
        return None, "grid must be a non-empty list of rows"
    if not all(isinstance(row, list) and row for row in value):
        return None, "grid rows must be non-empty lists"

    width = len(value[0])
    normalized = []
    for row in value:
        if len(row) != width:
            return None, "grid rows must all have the same width"
        normalized_row = []
        for cell in row:
            if not isinstance(cell, int) or isinstance(cell, bool) or not 0 <= cell <= 9:
                return None, "grid cells must be integer colors from 0 to 9"
            normalized_row.append(cell)
        normalized.append(normalized_row)
    return normalized, None


def _normalize_pair(record: dict[str, Any]) -> ArcPair:
    input_grid, input_error = _validate_grid(record.get("input"))
    output_grid, output_error = _validate_grid(record.get("output"))
    if input_error or output_error or input_grid is None or output_grid is None:
        raise ValueError(f"Invalid ARC pair: input={input_error}, output={output_error}")
    return ArcPair(input=input_grid, output=output_grid)


_VERSION_META = {
    "1": {"url": ARC_AGI1_ZIP_URL, "extracted_dir": "ARC-AGI-master"},
    "2": {"url": ARC_AGI2_ZIP_URL, "extracted_dir": "ARC-AGI-2-main"},
}


def _resolve_data_root(args) -> Path:
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser()
        if not data_dir.exists():
            raise FileNotFoundError(f"ARC-AGI data_dir does not exist: {data_dir}")
        return data_dir

    meta = _VERSION_META[args.version]
    cache_dir = Path(args.cache_dir).expanduser()
    data_root = cache_dir / meta["extracted_dir"] / "data"
    if data_root.exists():
        return data_root

    if not args.download_if_missing:
        raise FileNotFoundError(f"ARC-AGI-{args.version} cache is empty and download_if_missing=False: {data_root}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ARC-AGI-%s data into %s", args.version, cache_dir)
    with NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(meta["url"], tmp_path)
        with zipfile.ZipFile(tmp_path) as archive:
            archive.extractall(cache_dir)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Downloaded ARC-AGI-{args.version} archive did not contain expected data root: {data_root}"
        )
    return data_root


def _resolve_split_dir(data_root: Path, split: str) -> Path:
    if data_root.name in {"training", "evaluation"}:
        split_dir = data_root
    else:
        split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"ARC-AGI split directory does not exist: {split_dir}")
    return split_dir


def load_arc_agi_tasks(config: Args) -> list[ArcAgiTask]:
    arc_args = config.arc_agi_args
    data_root = _resolve_data_root(arc_args)
    split_dir = _resolve_split_dir(data_root, arc_args.split)
    paths = sorted(split_dir.glob("*.json"))
    if arc_args.max_tasks is not None:
        paths = paths[: int(arc_args.max_tasks)]

    tasks = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        train = [_normalize_pair(pair) for pair in record["train"]]
        test = [_normalize_pair(pair) for pair in record["test"]]
        tasks.append(
            ArcAgiTask(
                task_id=path.stem,
                split=arc_args.split,
                source_path=str(path),
                train=train,
                test=test,
            )
        )

    if not tasks:
        raise ValueError(f"No ARC-AGI tasks found in {split_dir}")
    return tasks


def _strip_markdown(text: str) -> str:
    text = text.strip()
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        text = answer_match.group(1).strip()
    attempt_match = re.search(r"<attempt>(.*?)</attempt>", text, re.DOTALL | re.IGNORECASE)
    if attempt_match:
        text = attempt_match.group(1).strip()
    fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_matches:
        text = fence_matches[-1].strip()
    return text


def _extract_candidate_outputs(action: str, num_tests: int) -> tuple[Optional[list[Any]], Optional[str]]:
    text = _strip_markdown(action)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"could not parse JSON: {exc.msg}"
    except RecursionError:
        return None, "could not parse JSON: nesting depth exceeded"

    if isinstance(parsed, dict):
        if "outputs" in parsed:
            outputs = parsed["outputs"]
        elif "answers" in parsed:
            outputs = parsed["answers"]
        elif "test" in parsed and isinstance(parsed["test"], list):
            outputs = [item.get("output") if isinstance(item, dict) else item for item in parsed["test"]]
        elif "output" in parsed:
            outputs = [parsed["output"]]
        else:
            return None, "JSON object must contain `output`, `outputs`, `answers`, or `test`"
    else:
        outputs = parsed

    if num_tests == 1 and _looks_like_grid(outputs):
        outputs = [outputs]

    if not isinstance(outputs, list):
        return None, "parsed answer must be a grid or a list of output grids"
    if len(outputs) != num_tests:
        return None, f"expected {num_tests} output grid(s), got {len(outputs)}"
    return outputs, None


def _looks_like_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) for row in value)
        and all(isinstance(cell, int) and not isinstance(cell, bool) for row in value for cell in row)
    )


def _grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def _count_cell_mismatches(candidate: list[list[int]], expected: list[list[int]]) -> int:
    if _grid_shape(candidate) != _grid_shape(expected):
        return 0
    return sum(
        int(c != e)
        for candidate_row, expected_row in zip(candidate, expected)
        for c, e in zip(candidate_row, expected_row)
    )


class ArcAgiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_name: str,
        task: str,
        config: Args,
        render_mode: Optional[str] = None,
        llm_judge=None,
    ):
        del render_mode, llm_judge
        super().__init__()
        self.env_name = env_name
        self.task = task
        self.config = config
        self.max_steps = int(config.arc_agi_args.max_turns)
        self.max_action_length = int(config.arc_agi_args.max_action_length)
        self.feedback_max_chars = int(config.arc_agi_args.feedback_max_chars)
        self.language_action_space = IdentityLanguageActionSpace(self.max_action_length)
        self.default_action = "[]"
        self.action_space = gym.spaces.Text(max_length=self.max_action_length)
        self.observation_space = gym.spaces.Dict({})
        self.actions = []
        self.dataset = load_arc_agi_tasks(config)
        self._episode_index = 0
        self._current_task: Optional[ArcAgiTask] = None
        self._attempts: list[dict[str, Any]] = []

    def get_wrapper_attr(self, name: str):
        return getattr(self, name)

    def get_text_action(self, action):
        return action

    def check_action_validity(self, candidate_action):
        return self.language_action_space.map(candidate_action), None

    def get_instruction_prompt(self, instructions=None):
        del instructions
        from feedback_bottleneck.envs.arc_agi.instruction_prompt import get_instruction_prompt

        return get_instruction_prompt(self.task)

    def get_stats(self):
        best_success = max((attempt["task_success"] for attempt in self._attempts), default=False)
        pass_at_2 = any(attempt["task_success"] for attempt in self._attempts[:2])
        stats = {
            "max_turns": self.max_steps,
            "num_attempts": len(self._attempts),
            "pass_at_2": float(pass_at_2),
            "best_success": float(best_success),
        }

        if self._current_task is not None:
            task = self._current_task
            num_test_pairs = len(task.test)
            first_test = task.test[0] if num_test_pairs > 0 else None
            test_input_height = len(first_test.input) if first_test else 0
            test_input_width = len(first_test.input[0]) if (first_test and first_test.input) else 0
            test_output_height = len(first_test.output) if first_test else 0
            test_output_width = len(first_test.output[0]) if (first_test and first_test.output) else 0

            size_changes = (test_input_height != test_output_height) or (test_input_width != test_output_width)
            max_dim = max(test_input_height, test_input_width, test_output_height, test_output_width)

            if not size_changes and max_dim <= 10:
                difficulty = "easy"
            elif size_changes and max_dim > 10:
                difficulty = "hard"
            else:
                difficulty = "medium"

            stats.update(
                {
                    "max_dim_grid_arc_agi": max_dim,
                    "difficulty": difficulty,
                }
            )

        return stats

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        super().reset(seed=seed)
        self._attempts = []
        if seed is None:
            idx = self._episode_index % len(self.dataset)
            self._episode_index += 1
        else:
            idx = int(seed) % len(self.dataset)
        self._current_task = self.dataset[idx]
        return self._build_observation(), {
            "problem_id": self._current_task.task_id,
            "source": self._current_task.source_path,
            "split": self._current_task.split,
            "correct": False,
        }

    def step(self, action: str):
        task = self._require_current_task()
        candidate_text = self.language_action_space.map(action)
        attempt = self._evaluate_attempt(candidate_text, task)
        self._attempts.append(attempt)

        solved = bool(attempt["task_success"])
        reward = 1.0 if solved else 0.0
        terminated = solved
        truncated = (not terminated) and len(self._attempts) >= self.max_steps
        best_success = max(attempt["task_success"] for attempt in self._attempts)
        pass_at_2 = any(previous_attempt["task_success"] for previous_attempt in self._attempts[:2])
        best_test_accuracy = max(attempt["test_accuracy"] for attempt in self._attempts)

        info = {
            "correct": solved,
            "problem_id": task.task_id,
            "source": task.source_path,
            "split": task.split,
            "attempts_used": len(self._attempts),
            "remaining_turns": max(self.max_steps - len(self._attempts), 0),
            "parse_valid": attempt["parse_valid"],
            "shape_valid": attempt["shape_valid"],
            "task_success": solved,
            "pass_at_2": float(pass_at_2),
            "test_accuracy": attempt["test_accuracy"],
            "best_test_accuracy": best_test_accuracy,
            "episode_extra_stats": {
                "correct": float(solved),
                "solved": float(solved),
                "task_success": float(solved),
                "pass_at_2": float(pass_at_2),
                "best_success": float(best_success),
                "parse_valid": float(attempt["parse_valid"]),
                "shape_valid": float(attempt["shape_valid"]),
                "test_accuracy": attempt["test_accuracy"],
                "best_test_accuracy": best_test_accuracy,
                "attempts_used": len(self._attempts),
                "remaining_turns": max(self.max_steps - len(self._attempts), 0),
            },
        }
        if terminated:
            info["end_status"] = "success"
        elif truncated:
            info["end_status"] = "max_turns"

        return self._build_observation(), reward, terminated, truncated, info

    def _evaluate_attempt(self, candidate_text: str, task: ArcAgiTask) -> dict[str, Any]:
        raw_outputs, parse_error = _extract_candidate_outputs(candidate_text, len(task.test))
        normalized_outputs = []
        validation_errors = []
        if raw_outputs is not None:
            for idx, output in enumerate(raw_outputs):
                grid, error = _validate_grid(output)
                if error:
                    validation_errors.append(f"test {idx}: {error}")
                    normalized_outputs.append(None)
                else:
                    normalized_outputs.append(grid)

        parse_valid = parse_error is None
        shape_valid = bool(parse_valid and not validation_errors)
        correct_outputs = 0
        shape_mismatches = 0
        cell_mismatches = 0
        if shape_valid:
            for candidate, pair in zip(normalized_outputs, task.test):
                if candidate == pair.output:
                    correct_outputs += 1
                elif _grid_shape(candidate) != _grid_shape(pair.output):
                    shape_mismatches += 1
                else:
                    cell_mismatches += _count_cell_mismatches(candidate, pair.output)

        task_success = bool(shape_valid and correct_outputs == len(task.test))
        feedback = self._build_feedback(
            parse_error=parse_error,
            validation_errors=validation_errors,
            correct_outputs=correct_outputs,
            total_outputs=len(task.test),
            shape_mismatches=shape_mismatches,
            cell_mismatches=cell_mismatches,
            task_success=task_success,
        )
        return {
            "turn": len(self._attempts) + 1,
            "answer": candidate_text,
            "parse_valid": parse_valid,
            "shape_valid": shape_valid,
            "task_success": task_success,
            "correct_outputs": correct_outputs,
            "total_outputs": len(task.test),
            "test_accuracy": correct_outputs / max(len(task.test), 1),
            "shape_mismatches": shape_mismatches,
            "cell_mismatches": cell_mismatches,
            "feedback": feedback[: self.feedback_max_chars],
        }

    def _build_feedback(
        self,
        *,
        parse_error: Optional[str],
        validation_errors: list[str],
        correct_outputs: int,
        total_outputs: int,
        shape_mismatches: int,
        cell_mismatches: int,
        task_success: bool,
    ) -> str:
        if task_success:
            return "Correct: every predicted test output grid matches the reference output."
        if parse_error:
            return (
                "The answer could not be parsed as ARC output JSON. "
                f"Parser error: {parse_error}. Return only a JSON grid or a JSON object with an `outputs` list."
            )
        if validation_errors:
            return "The parsed answer is not a valid ARC grid output. " + "; ".join(validation_errors)

        parts = [f"{correct_outputs}/{total_outputs} test output grid(s) matched exactly."]
        if shape_mismatches:
            parts.append(f"{shape_mismatches} output grid(s) had the wrong shape.")
        if cell_mismatches:
            parts.append(
                "For outputs with the correct shape, at least one cell color or position is still wrong. "
                "Re-check the transformation from the training examples rather than copying local patterns."
            )
        return " ".join(parts)

    def _build_observation(self) -> dict[str, Any]:
        task = self._require_current_task()
        attempts_text = self._format_attempts()
        latest_attempt = self._attempts[-1] if self._attempts else None
        last_feedback = latest_attempt["feedback"] if latest_attempt else "No ARC feedback yet."
        remaining_turns = self.max_steps - len(self._attempts)
        train_pairs = [{"input": pair.input, "output": pair.output} for pair in task.train]
        test_inputs = [pair.input for pair in task.test]
        expected_outputs = [pair.output for pair in task.test]
        problem_text = json.dumps(
            {"train": train_pairs, "test": [{"input": grid} for grid in test_inputs]},
            ensure_ascii=True,
            separators=(",", ":"),
        )

        return {
            "obs": {
                "problem": problem_text,
                "problem_id": task.task_id,
                "source": task.source_path,
                "split": task.split,
                "train": train_pairs,
                "test_inputs": test_inputs,
                "answer": json.dumps(expected_outputs, ensure_ascii=True, separators=(",", ":")),
                "attempts": list(self._attempts),
                "latest_attempt": dict(latest_attempt) if latest_attempt else None,
                "attempts_remaining": remaining_turns,
                "num_attempts": len(self._attempts),
                "last_feedback": last_feedback,
            },
            "text": {
                "short_term_context": last_feedback,
                "long_term_context": (
                    f"ARC-AGI task id: {task.task_id}\n"
                    f"Split: {task.split}\n"
                    f"Task JSON without test outputs:\n{problem_text}\n"
                    f"Attempts remaining: {remaining_turns}\n"
                    f"Previous attempts:\n{attempts_text}"
                ),
            },
        }

    def _format_attempts(self) -> str:
        if not self._attempts:
            return "No previous attempts."
        lines = []
        for attempt in self._attempts:
            lines.append(
                f"Attempt {attempt['turn']}: success={attempt['task_success']} "
                f"parse_valid={attempt['parse_valid']} shape_valid={attempt['shape_valid']} "
                f"exact_outputs={attempt['correct_outputs']}/{attempt['total_outputs']}"
            )
        return "\n".join(lines)

    def _require_current_task(self) -> ArcAgiTask:
        if self._current_task is None:
            raise RuntimeError("Environment must be reset before stepping.")
        return self._current_task


def make_arc_agi_env(env_name, task, config, render_mode: Optional[str] = None, llm_judge=None):
    env = ArcAgiEnv(env_name, task, config, render_mode=render_mode, llm_judge=llm_judge)
    return EnvWrapper(env, env_name, task, args=config)
