import hashlib
from typing import Optional

from feedback_bottleneck.envs.dataset_utils import filter_dataset_by_problem_split as _filter_dataset_by_problem_split
from feedback_bottleneck.envs.dataset_utils import split_problem_ids as _split_problem_ids


def _normalize_problem_text(problem: str) -> str:
    return " ".join(str(problem).split())


def make_problem_id(problem: str) -> str:
    normalized_problem = _normalize_problem_text(problem)
    digest = hashlib.sha1(normalized_problem.encode("utf-8")).hexdigest()[:16]
    return f"problem_{digest}"


def get_problem_ids(dataset) -> list[str]:
    if "problem_id" in dataset.column_names:
        return [str(problem_id) for problem_id in dataset["problem_id"]]

    if "problem" not in dataset.column_names:
        raise ValueError(
            "Math problem splitting requires either a `problem_id` column or a `problem` column. "
            f"Available columns: {dataset.column_names}"
        )

    return [make_problem_id(problem) for problem in dataset["problem"]]


def split_problem_ids(
    problem_ids: list[str],
    *,
    split_seed: int,
    test_size: int,
) -> tuple[set[str], set[str]]:
    return _split_problem_ids(
        problem_ids,
        split_seed=split_seed,
        test_size=test_size,
        split_label="math",
    )


def filter_dataset_by_problem_split(
    dataset,
    *,
    split_name: str,
    split_seed: int,
    test_size: Optional[int],
):
    problem_ids = get_problem_ids(dataset)
    return _filter_dataset_by_problem_split(
        dataset,
        problem_ids=problem_ids,
        split_name=split_name,
        split_seed=split_seed,
        test_size=test_size,
        split_label="math",
    )
