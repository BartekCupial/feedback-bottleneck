import json
import logging
import pprint
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tyro
from datasets import Array2D, Array3D, Dataset, Features, Image, Sequence, Value


from feedback_bottleneck.config.args import Args
from feedback_bottleneck.llm.actor import create_llm_actors
from feedback_bottleneck.llm.agent import AgentFactory
from feedback_bottleneck.llm.evaluator import Evaluator
from feedback_bottleneck.utils.math_eval_metrics import build_cumulative_accuracy_metrics, build_cumulative_accuracy_wandb_payload
from feedback_bottleneck.utils.wandb_utils import finish_wandb, init_wandb, log_wandb_html_trajectories


def _get_llm_metadata(args: Args) -> Dict[str, str]:
    # API (OpenAI/Anthropic etc.)
    if getattr(args, "actor_type", "vllm") == "client":
        backend = str(getattr(args, "client_name", "") or "")
        model_id = str(getattr(args, "client_model_id", "") or "")
        base_url = str(getattr(args, "client_base_url", "") or "")
        tokenizer_id = str(getattr(args, "tokenizer_id", "") or "")
    else:
        # local (vLLM)
        backend = "local"
        model_id = str(getattr(args, "model_id", "") or "")
        base_url = ""
        tokenizer_id = str(getattr(args, "tokenizer_id", "") or "")

    return {
        "llm_backend": backend,
        "llm_model_id": model_id,
        "llm_tokenizer_id": tokenizer_id,
        "llm_base_url": base_url,
    }


def infer_obs_features(episode_df):
    """
    Dynamically creates efficient Arrow features (Array2D, Array3D) for
    numpy array columns to avoid inefficient list-of-lists storage.
    """
    custom_features = {}

    for col in episode_df.columns:
        # We only look at the new observation columns we created
        if not col.startswith("obs_"):
            continue

        # Peek at the first non-None value to determine shape and dtype
        valid_rows = episode_df[col].dropna()
        if valid_rows.empty:
            continue

        first_item = valid_rows.iloc[0]

        if isinstance(first_item, np.ndarray):
            shape = first_item.shape
            dtype = str(first_item.dtype)

            # NLE Glyphs (2D) -> Array2D
            if len(shape) == 2:
                custom_features[col] = Array2D(shape=shape, dtype=dtype)
            # Crafter Images (3D) -> Array3D
            elif len(shape) == 3:
                custom_features[col] = Array3D(shape=shape, dtype=dtype)
            # BLStats (1D) -> Sequence
            elif len(shape) == 1:
                custom_features[col] = Sequence(Value(dtype))

    return custom_features


def _get_representative_row_for_features(episode_df: pd.DataFrame) -> dict:
    """
    Pick one representative value per column for schema inference.

    Using only the first row is brittle for columns that start with empty lists
    or nulls and later contain structured data, such as OmniMath attempts.
    """
    representative_row = {}

    for col in episode_df.columns:
        series = episode_df[col]
        chosen = None

        for value in series:
            if value is None:
                continue
            if isinstance(value, float) and pd.isna(value):
                continue

            # Prefer a non-empty list so HF infers the element type instead of
            # defaulting to Sequence(null).
            if isinstance(value, list):
                if len(value) == 0:
                    if chosen is None:
                        chosen = value
                    continue
                chosen = value
                break

            chosen = value
            break

        representative_row[col] = chosen

    return representative_row


def _jsonify_nested_obs_columns(episode_df: pd.DataFrame) -> pd.DataFrame:
    nested_columns = [
        col
        for col in episode_df.columns
        if col.startswith("obs_") and episode_df[col].map(lambda value: isinstance(value, (dict, list))).any()
    ]
    if not nested_columns:
        return episode_df

    episode_df = episode_df.copy()
    for col in nested_columns:
        episode_df[col] = episode_df[col].apply(
            lambda value: json.dumps(value, ensure_ascii=True, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )
    return episode_df


def create_huggingface_dataset(episode_df, args: Args):
    episode_df = _jsonify_nested_obs_columns(episode_df)

    # Use representative non-null/non-empty values per column for robust schema
    # inference. First-row inference breaks when row 0 contains placeholders
    # like [] for columns that later hold structs.
    representative_row = _get_representative_row_for_features(episode_df)
    base_features = Dataset.from_dict({k: [v] for k, v in representative_row.items()}).features.copy()

    # infer and update the types of the observation columns (starting with 'obs_')
    obs_features = infer_obs_features(episode_df)
    base_features.update(obs_features)

    # persist full environment states in the dataset, when using the
    # `crafter_with_state_saving` fork, each step has a `state_dumps`
    # entry containing a compressed joblib dump of the environment state.
    # Casting this column to Binary enables HuggingFace to store and
    # reload the bytes without mangling, with these state dumps we can
    # later restore any saved state and resume rollouts from arbitrary
    # points in an episode.
    if "image_paths" in episode_df.columns:
        base_features["image_paths"] = Image()
    if "state_dumps" in episode_df.columns:
        base_features["state_dumps"] = Value("binary")

    # create the dataset using the new schema
    data_dict = episode_df.to_dict(orient="list")
    dataset = Dataset.from_dict(data_dict, features=base_features)

    episode_id_to_index = {eid: idx for idx, eid in enumerate(dataset.unique("episode_id"))}
    episode_indices = [episode_id_to_index[eid] for eid in dataset["episode_id"]]
    dataset = dataset.add_column("episode_index", episode_indices)

    llm_metadata = _get_llm_metadata(args)
    n = len(dataset)
    for k, v in llm_metadata.items():
        dataset = dataset.add_column(k, [v] * n)

    return dataset


def collect_episodes(args: Args):
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s|  %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Arguments:\n%s", pprint.pformat(vars(args)))

    llm_actor, llm_judge = create_llm_actors(args)
    agent_factory = AgentFactory(args, llm_actor, llm_judge)

    evaluator = Evaluator(agent_factory, args)

    try:
        results = evaluator.collect_episodes(args.num_eval_episodes)
        episode_dfs, episode_logs = zip(*results)
        episode_df = pd.concat(episode_dfs, ignore_index=True)
        dataset = create_huggingface_dataset(episode_df, args)

        # Optionally save dataset to disk
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(output_dir)
            print(f"Dataset saved locally at {output_dir}")

        # Optionally push to Hugging Face Hub
        if args.push_to_hub and args.hub_repo:
            print(f"Pushing to Hugging Face Hub: {args.hub_repo}")
            dataset.push_to_hub(args.hub_repo)

        init_wandb(args)
        trigger_sync = None


        if args.log_wandb:
            import wandb

            episode_log = pd.DataFrame(episode_logs)
            wandb.log({"collect/episode_log": wandb.Table(dataframe=episode_log)})
            episode_desc = episode_log.describe().reset_index()
            wandb.log({"collect/episode_log_describe": wandb.Table(dataframe=episode_desc)})
            cumulative_metrics = build_cumulative_accuracy_metrics(args, episode_logs, prefix="collect")
            if cumulative_metrics:
                wandb.log(cumulative_metrics)
            cumulative_payload = build_cumulative_accuracy_wandb_payload(args, episode_logs, prefix="collect")
            if cumulative_payload:
                wandb.log(cumulative_payload)

            print("Generating interactive trajectory visualizations...")
            log_wandb_html_trajectories(episode_df, num_episodes=5)

        if trigger_sync is not None:
            trigger_sync()
        finish_wandb(args)
    finally:
        # Clean up resources
        evaluator.shutdown()


if __name__ == "__main__":
    args = tyro.cli(Args)
    collect_episodes(args)
