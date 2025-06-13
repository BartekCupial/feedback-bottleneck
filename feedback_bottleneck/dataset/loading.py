import glob
import logging
import os
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def resolve_hf_snapshot_path(repo_id: str, repo_type: str = "dataset") -> Optional[str]:
    if "/" not in repo_id:
        return None

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    owner, name = repo_id.split("/", 1)
    repo_cache_dir = os.path.join(hf_home, "hub", f"{repo_type}s--{owner}--{name}")

    if not os.path.isdir(repo_cache_dir):
        return None

    ref_path = os.path.join(repo_cache_dir, "refs", "main")
    if os.path.isfile(ref_path):
        with open(ref_path) as f:
            revision = f.read().strip()
        snapshot_path = os.path.join(repo_cache_dir, "snapshots", revision)
        if os.path.isdir(snapshot_path):
            return snapshot_path

    snapshots_dir = os.path.join(repo_cache_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    snapshot_names = sorted(os.listdir(snapshots_dir))
    if not snapshot_names:
        return None

    return os.path.join(snapshots_dir, snapshot_names[-1])


def _infer_local_dataset_loader(path: str, split: Optional[str]):
    path_obj = Path(path).expanduser()

    if path_obj.is_dir():
        try:
            return load_from_disk(str(path_obj))
        except Exception:
            arrow_files = glob.glob(os.path.join(str(path_obj), "**/*.arrow"), recursive=True)
            parquet_files = glob.glob(os.path.join(str(path_obj), "**/*.parquet"), recursive=True)

            if arrow_files:
                return load_dataset("arrow", data_files=arrow_files, split=split)
            if parquet_files:
                return load_dataset("parquet", data_files=parquet_files, split=split)

            raise ValueError(f"No dataset files found in {path_obj}")

    suffix = path_obj.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path_obj), split=split or "train")
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path_obj), split=split or "train")
    if suffix == ".arrow":
        return load_dataset("arrow", data_files=str(path_obj), split=split or "train")

    raise FileNotFoundError(f"Unsupported local dataset path: {path_obj}")


def load_dataset_any(
    dataset_path: str,
    *,
    name: Optional[str] = None,
    split: Optional[str] = "train",
):
    expanded_path = str(Path(dataset_path).expanduser())

    if os.path.exists(expanded_path):
        return _infer_local_dataset_loader(expanded_path, split)

    try:
        return load_dataset(dataset_path, name=name, split=split)
    except ConnectionError:
        snapshot_path = resolve_hf_snapshot_path(dataset_path, repo_type="dataset")
        if snapshot_path is None:
            raise
        logging.info("Falling back to cached Hugging Face dataset snapshot for %s at %s", dataset_path, snapshot_path)
        return _infer_local_dataset_loader(snapshot_path, split)


def ensure_dataset_split(obj, split: str) -> Dataset:
    if isinstance(obj, Dataset):
        return obj

    if isinstance(obj, DatasetDict):
        if split in obj:
            return obj[split]
        if "train" in obj:
            return obj["train"]
        first_split = next(iter(obj.keys()))
        return obj[first_split]

    raise TypeError(f"Unsupported dataset object type: {type(obj)}")


def load_single_split_dataset(
    dataset_path: str,
    *,
    name: Optional[str] = None,
    split: str = "train",
) -> Dataset:
    return ensure_dataset_split(load_dataset_any(dataset_path, name=name, split=split), split)
