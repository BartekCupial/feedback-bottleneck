import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TypeVar

import numpy as np

from feedback_bottleneck.dataset.loading import load_single_split_dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetExample:
    problem: str
    answer: str
    source: str
    problem_id: str
    raw_record: Optional[dict[str, Any]] = None


ExampleT = TypeVar("ExampleT", bound=DatasetExample)


def split_problem_ids(
    problem_ids: Sequence[str],
    *,
    split_seed: int,
    test_size: Optional[int],
    split_label: str,
) -> tuple[set[str], set[str]]:
    unique_problem_ids = sorted(set(problem_ids))
    if not unique_problem_ids:
        raise ValueError(f"Cannot derive a {split_label} problem split from an empty problem_id set.")

    if test_size is None:
        raise ValueError(f"{split_label} problem split requires a non-null test_size.")

    if not 0 < test_size < len(unique_problem_ids):
        raise ValueError(
            f"Resolved {split_label} problem test split must be between 1 and total_unique_problem_ids - 1, "
            f"got {test_size} for {len(unique_problem_ids)} unique problems."
        )

    rng = np.random.default_rng(split_seed)
    permuted = list(rng.permutation(unique_problem_ids))
    test_problem_ids = set(permuted[:test_size])
    train_problem_ids = set(permuted[test_size:])

    if train_problem_ids & test_problem_ids:
        raise ValueError(f"{split_label} problem split produced overlapping train/test problem_ids.")

    return train_problem_ids, test_problem_ids


def _select_problem_indices(
    problem_ids: Sequence[str],
    *,
    split_name: str,
    split_seed: int,
    test_size: Optional[int],
    split_label: str,
) -> tuple[list[int], set[str]]:
    if split_name == "all":
        return list(range(len(problem_ids))), set(problem_ids)

    train_problem_ids, test_problem_ids = split_problem_ids(
        problem_ids,
        split_seed=split_seed,
        test_size=test_size,
        split_label=split_label,
    )

    selected_problem_ids = train_problem_ids if split_name == "train" else test_problem_ids
    selected_indices = [idx for idx, problem_id in enumerate(problem_ids) if problem_id in selected_problem_ids]

    if not selected_indices:
        raise ValueError(f"{split_label} problem split {split_name!r} produced an empty dataset.")

    return selected_indices, selected_problem_ids


def filter_items_by_problem_split(
    items: Sequence[ExampleT],
    *,
    split_name: str,
    split_seed: int,
    test_size: Optional[int],
    split_label: str,
    problem_id_getter: Optional[Callable[[ExampleT], str]] = None,
) -> list[ExampleT]:
    if split_name == "all":
        return list(items)

    getter = problem_id_getter or (lambda item: item.problem_id)
    problem_ids = [getter(item) for item in items]
    selected_indices, selected_problem_ids = _select_problem_indices(
        problem_ids,
        split_name=split_name,
        split_seed=split_seed,
        test_size=test_size,
        split_label=split_label,
    )
    filtered_items = [items[idx] for idx in selected_indices]

    logger.info(
        "Applied %s problem split=%s with %s rows across %s unique problems (seed=%s).",
        split_label,
        split_name,
        len(filtered_items),
        len(selected_problem_ids),
        split_seed,
    )
    return filtered_items


def filter_dataset_by_problem_split(
    dataset,
    *,
    problem_ids: Sequence[str],
    split_name: str,
    split_seed: int,
    test_size: Optional[int],
    split_label: str,
):
    if split_name == "all":
        return dataset

    selected_indices, selected_problem_ids = _select_problem_indices(
        problem_ids,
        split_name=split_name,
        split_seed=split_seed,
        test_size=test_size,
        split_label=split_label,
    )
    filtered_dataset = dataset.select(selected_indices)
    logger.info(
        "Applied %s problem split=%s with %s rows across %s unique problems (seed=%s).",
        split_label,
        split_name,
        len(filtered_dataset),
        len(selected_problem_ids),
        split_seed,
    )
    return filtered_dataset


def load_dataset_examples(
    dataset_path: str,
    dataset_config: Optional[str],
    dataset_split: str,
    *,
    normalize_record: Callable[[dict[str, Any], int], ExampleT],
    problem_split: str,
    problem_split_seed: int,
    problem_split_test_size: Optional[int],
    split_label: str,
    dataset_filter: Optional[Callable[[Any], Any]] = None,
    empty_error_message: Optional[str] = None,
) -> list[ExampleT]:
    dataset = load_single_split_dataset(
        dataset_path,
        name=dataset_config,
        split=dataset_split,
    )
    logger.info("Loaded %s dataset with %s samples", split_label, len(dataset))

    if dataset_filter is not None:
        dataset = dataset_filter(dataset)

    examples = [normalize_record(dict(row), idx) for idx, row in enumerate(dataset)]
    if not examples:
        raise ValueError(empty_error_message or f"No {split_label} examples found in {dataset_path}.")

    return filter_items_by_problem_split(
        examples,
        split_name=problem_split,
        split_seed=problem_split_seed,
        test_size=problem_split_test_size,
        split_label=split_label,
    )
