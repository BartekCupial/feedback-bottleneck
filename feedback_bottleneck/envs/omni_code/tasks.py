from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")


CodeTestKind = Literal["stdin_stdout", "call_based", "asserts"]


@dataclass
class CodeTestSpec:
    kind: CodeTestKind
    inputs: list[Any] = field(default_factory=list)
    outputs: list[Any] = field(default_factory=list)
    fn_name: Optional[str] = None
    asserts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CodeTask:
    task_id: str
    prompt: str
    starter_code: str
    reference_solution: str
    tests: CodeTestSpec
    source: str = ""
    difficulty: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "starter_code": self.starter_code,
            "reference_solution": self.reference_solution,
            "tests": self.tests.to_dict(),
            "source": self.source,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }


def _strip_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_problem_text(problem: str) -> str:
    return " ".join(str(problem).split())


def make_task_id(prompt: str, source: str = "") -> str:
    normalized_problem = _normalize_problem_text(prompt)
    if source:
        normalized_problem = f"{source}\n{normalized_problem}"
    digest = hashlib.sha1(normalized_problem.encode("utf-8")).hexdigest()[:16]
    return f"task_{digest}"


def parse_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return value


def _first_nonempty_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, list):
            for item in value:
                text = _strip_text(item)
                if text:
                    return text
            continue

        parsed = parse_jsonish(value)
        if isinstance(parsed, list):
            for item in parsed:
                text = _strip_text(item)
                if text:
                    return text
            continue

        text = _strip_text(parsed)
        if text:
            return text
    return ""


def _normalize_stdio_cases(cases: list[dict[str, Any]]) -> Optional[CodeTestSpec]:
    inputs = []
    outputs = []
    for case in cases:
        inp = case.get("input") if isinstance(case, dict) else None
        out = case.get("output") if isinstance(case, dict) else None
        if inp is None or out is None:
            return None
        inputs.append(inp)
        outputs.append(out)
    if not inputs:
        return None
    return CodeTestSpec(kind="stdin_stdout", inputs=inputs, outputs=outputs)


def _normalize_examples_blob(blob: Any) -> list[dict[str, str]]:
    parsed = parse_jsonish(blob)
    if not isinstance(parsed, list):
        return []

    normalized = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        inp = _strip_text(item.get("input"))
        out = _strip_text(item.get("output"))
        if not inp or not out:
            continue
        normalized.append({"input": inp, "output": out})
    return normalized


def _normalize_constraints(record: dict[str, Any]) -> str:
    constraints = _strip_text(record.get("constraints"))
    if constraints:
        return constraints

    parts = []
    time_limit = record.get("time_limit")
    memory_limit = record.get("memory_limit")
    if time_limit not in (None, ""):
        parts.append(f"Time limit: {time_limit} s")
    if memory_limit not in (None, ""):
        parts.append(f"Memory limit: {memory_limit} MB")
    return "\n".join(parts).strip()


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None

    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
        return parsed if np.isfinite(parsed) else None
    except ValueError:
        match = _FLOAT_RE.search(text.replace(",", "."))
        if match is None:
            return None
        parsed = float(match.group(0))
        return parsed if np.isfinite(parsed) else None


def _count_test_cases(spec: CodeTestSpec) -> int:
    if spec.kind == "asserts":
        return len(spec.asserts)
    return max(len(spec.inputs), len(spec.outputs))


def resolve_execution_timeouts(
    task: CodeTask,
    *,
    default_case_timeout_s: float,
    default_task_timeout_s: Optional[float] = None,
    use_dataset_time_limit: bool = False,
    time_limit_multiplier: float = 10.0,
    min_case_timeout_s: Optional[float] = None,
    max_case_timeout_s: Optional[float] = None,
    task_timeout_case_budget_multiplier: float = 2.0,
) -> tuple[float, Optional[float]]:
    case_timeout_s = float(default_case_timeout_s)

    if use_dataset_time_limit:
        dataset_time_limit_s = _parse_optional_float(
            task.metadata.get("time_limit_seconds", task.metadata.get("time_limit"))
        )
        if dataset_time_limit_s is not None and dataset_time_limit_s > 0:
            case_timeout_s = dataset_time_limit_s * float(time_limit_multiplier)

    if min_case_timeout_s is not None:
        case_timeout_s = max(case_timeout_s, float(min_case_timeout_s))
    if max_case_timeout_s is not None:
        case_timeout_s = min(case_timeout_s, float(max_case_timeout_s))

    task_timeout_s = None if default_task_timeout_s is None else float(default_task_timeout_s)
    if task_timeout_s is None:
        num_cases = max(_count_test_cases(task.tests), 1)
        task_timeout_s = case_timeout_s * num_cases * float(task_timeout_case_budget_multiplier)

    task_timeout_s = max(task_timeout_s, case_timeout_s)
    return case_timeout_s, task_timeout_s


def _normalize_tests_blob(blob: Any, max_cases: Optional[int] = None) -> Optional[CodeTestSpec]:
    parsed = parse_jsonish(blob)

    if isinstance(parsed, dict):
        if "asserts" in parsed and parsed["asserts"]:
            asserts = [str(item) for item in parsed["asserts"] if str(item).strip()]
            if not asserts:
                return None
            if max_cases is not None:
                asserts = asserts[:max_cases]
            return CodeTestSpec(kind="asserts", asserts=asserts)

        if "test_list" in parsed and parsed["test_list"]:
            asserts = [str(item) for item in parsed["test_list"] if str(item).strip()]
            if not asserts:
                return None
            if max_cases is not None:
                asserts = asserts[:max_cases]
            return CodeTestSpec(kind="asserts", asserts=asserts)

        inputs = parsed.get("inputs")
        outputs = parsed.get("outputs")
        fn_name = _strip_text(parsed.get("fn_name") or parsed.get("function_name"))
        if isinstance(inputs, list) and isinstance(outputs, list) and len(inputs) == len(outputs) and inputs:
            if max_cases is not None:
                inputs = inputs[:max_cases]
                outputs = outputs[:max_cases]
            return CodeTestSpec(
                kind="call_based" if fn_name else "stdin_stdout",
                inputs=list(inputs),
                outputs=list(outputs),
                fn_name=fn_name or None,
            )

        public_tests = parsed.get("public_tests")
        private_tests = parsed.get("private_tests")
        if isinstance(public_tests, list) or isinstance(private_tests, list):
            merged = []
            if isinstance(public_tests, list):
                merged.extend(public_tests)
            if isinstance(private_tests, list):
                merged.extend(private_tests)
            if max_cases is not None:
                merged = merged[:max_cases]
            return _normalize_stdio_cases(merged)

        official_tests = parsed.get("official_tests")
        if isinstance(official_tests, list):
            if max_cases is not None:
                official_tests = official_tests[:max_cases]
            return _normalize_stdio_cases(official_tests)

    if isinstance(parsed, list):
        if parsed and all(isinstance(item, str) for item in parsed):
            asserts = [str(item) for item in parsed if str(item).strip()]
            if not asserts:
                return None
            if max_cases is not None:
                asserts = asserts[:max_cases]
            return CodeTestSpec(kind="asserts", asserts=asserts)

        if parsed and all(isinstance(item, dict) for item in parsed):
            if max_cases is not None:
                parsed = parsed[:max_cases]
            return _normalize_stdio_cases(parsed)

    return None


def record_matches_code_filters(
    record: dict[str, Any],
    *,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    require_official_tests_complete: bool = False,
    require_stdio: bool = False,
    require_no_generated_checker: bool = False,
    required_tags: Optional[list[str] | tuple[str, ...]] = None,
) -> bool:
    rating = record.get("rating", record.get("cf_rating"))
    parsed_rating = _parse_optional_float(rating)
    if min_rating is not None and parsed_rating is not None and parsed_rating < int(min_rating):
        return False
    if max_rating is not None and parsed_rating is not None and parsed_rating > int(max_rating):
        return False

    if require_official_tests_complete and record.get("official_tests_complete") is not True:
        return False

    if require_stdio:
        input_mode = str(record.get("input_mode", "")).strip().lower()
        if input_mode and input_mode != "stdio":
            return False

    if require_no_generated_checker and str(record.get("generated_checker") or "").strip():
        return False

    if required_tags:
        row_tags = {str(tag).strip().lower() for tag in (record.get("tags") or [])}
        wanted = {str(tag).strip().lower() for tag in required_tags}
        if not wanted.issubset(row_tags):
            return False

    return True


def normalize_task_record(
    record: dict[str, Any],
    idx: int,
    *,
    source_name: str = "",
    max_cases: Optional[int] = None,
    require_reference_solution: bool = True,
) -> Optional[CodeTask]:
    problem_text = _first_nonempty_text(
        record.get("description"),
        record.get("problem_statement"),
        record.get("statement"),
        record.get("problem"),
        record.get("question"),
        record.get("prompt"),
    )
    input_format = _first_nonempty_text(record.get("input"), record.get("input_format"))
    output_format = _first_nonempty_text(record.get("output"), record.get("output_format"))
    interaction_format = _first_nonempty_text(record.get("interaction_format"))
    note = _first_nonempty_text(record.get("note"))
    constraints = _normalize_constraints(record)
    examples = _normalize_examples_blob(record.get("examples"))

    prompt_sections = [problem_text]
    if input_format:
        prompt_sections.append(f"Input format:\n{input_format}")
    if output_format:
        prompt_sections.append(f"Output format:\n{output_format}")
    if interaction_format:
        prompt_sections.append(f"Interaction format:\n{interaction_format}")
    if constraints:
        prompt_sections.append(f"Constraints:\n{constraints}")
    if note:
        prompt_sections.append(f"Note:\n{note}")
    if examples:
        example_blocks = [
            f"Example input:\n{example['input']}\n\nExample output:\n{example['output']}" for example in examples[:2]
        ]
        prompt_sections.append("\n\n".join(example_blocks))

    prompt = "\n\n".join(section for section in prompt_sections if section)
    prompt = _first_nonempty_text(
        prompt,
        record.get("prompt"),
        record.get("question"),
        record.get("problem"),
        record.get("problem_statement"),
        record.get("statement"),
        record.get("description"),
    )
    starter_code = _first_nonempty_text(
        record.get("starter_code"),
        record.get("starter_code_py"),
        record.get("code_prompt"),
    )
    reference_solution = _first_nonempty_text(
        record.get("canonical_solution"),
        record.get("reference_solution"),
        record.get("solution"),
        record.get("solutions"),
    )
    tests = (
        _normalize_tests_blob(record.get("tests"), max_cases=max_cases)
        or _normalize_tests_blob(record.get("input_output"), max_cases=max_cases)
        or _normalize_tests_blob(record.get("test_cases"), max_cases=max_cases)
        or _normalize_tests_blob({"official_tests": record.get("official_tests")}, max_cases=max_cases)
        or _normalize_tests_blob(
            {
                "public_tests": record.get("public_tests"),
                "private_tests": record.get("private_tests"),
            },
            max_cases=max_cases,
        )
    )

    source = _first_nonempty_text(record.get("source"), record.get("dataset"), source_name)
    difficulty = _first_nonempty_text(
        record.get("difficulty"),
        record.get("cf_rating"),
        record.get("rating"),
        record.get("level"),
    )
    task_id = _first_nonempty_text(record.get("task_id"), record.get("problem_id"))
    if not task_id and prompt:
        task_id = make_task_id(prompt, source)

    if not prompt or tests is None or (require_reference_solution and not reference_solution):
        return None

    metadata = dict(record)
    metadata["dataset_index"] = idx
    if input_format and "input" not in metadata:
        metadata["input"] = input_format
    if output_format and "output" not in metadata:
        metadata["output"] = output_format
    if constraints and "constraints" not in metadata:
        metadata["constraints"] = constraints
    if examples and "example" not in metadata:
        metadata["example"] = examples[0]
        metadata["examples_normalized"] = examples
    time_limit_seconds = _parse_optional_float(record.get("time_limit"))
    if time_limit_seconds is not None and "time_limit_seconds" not in metadata:
        metadata["time_limit_seconds"] = time_limit_seconds
    memory_limit_mb = _parse_optional_float(record.get("memory_limit"))
    if memory_limit_mb is not None and "memory_limit_mb" not in metadata:
        metadata["memory_limit_mb"] = memory_limit_mb

    return CodeTask(
        task_id=task_id,
        prompt=prompt,
        starter_code=starter_code,
        reference_solution=reference_solution,
        tests=tests,
        source=source,
        difficulty=difficulty,
        metadata=metadata,
    )


def split_problem_ids(
    problem_ids: list[str],
    *,
    split_seed: int,
    test_size: int,
) -> tuple[set[str], set[str]]:
    unique_problem_ids = sorted(set(problem_ids))
    if not unique_problem_ids:
        raise ValueError("Cannot derive a code problem split from an empty task_id set.")

    if not 0 < test_size < len(unique_problem_ids):
        raise ValueError(
            "Resolved code problem test split must be between 1 and total_unique_problem_ids - 1, "
            f"got {test_size} for {len(unique_problem_ids)} unique problems."
        )

    rng = np.random.default_rng(split_seed)
    permuted = list(rng.permutation(unique_problem_ids))
    test_problem_ids = set(permuted[:test_size])
    train_problem_ids = set(permuted[test_size:])

    if train_problem_ids & test_problem_ids:
        raise ValueError("Code problem split produced overlapping train/test task_ids.")

    return train_problem_ids, test_problem_ids


def filter_dataset_by_problem_split(
    dataset,
    *,
    split_name: str,
    split_seed: int,
    test_size: Optional[int],
):
    if split_name == "all":
        return dataset

    if test_size is None:
        raise ValueError("code_args.problem_split_test_size must be set when using a code problem split.")

    if isinstance(dataset, list):
        problem_ids = [str(row["task_id"]) for row in dataset]
    else:
        problem_ids = [str(task_id) for task_id in dataset["task_id"]]
    train_problem_ids, test_problem_ids = split_problem_ids(
        problem_ids,
        split_seed=split_seed,
        test_size=test_size,
    )

    selected_problem_ids = train_problem_ids if split_name == "train" else test_problem_ids
    selected_indices = [idx for idx, problem_id in enumerate(problem_ids) if problem_id in selected_problem_ids]

    if not selected_indices:
        raise ValueError(f"Code problem split {split_name!r} produced an empty dataset.")

    if isinstance(dataset, list):
        filtered_dataset = [dataset[idx] for idx in selected_indices]
    else:
        filtered_dataset = dataset.select(selected_indices)
    logger.info(
        "Applied code problem split=%s with %s rows across %s unique tasks (seed=%s).",
        split_name,
        len(filtered_dataset),
        len(selected_problem_ids),
        split_seed,
    )
    return filtered_dataset
