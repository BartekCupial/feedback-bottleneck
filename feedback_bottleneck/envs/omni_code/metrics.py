from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_EPISODE_COLUMNS = ("episode_id", "obs_task_id", "timestep", "rewards", "terms")


def _validate_episode_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_EPISODE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required episode columns: {missing}")


def _normalize_ks(ks: Iterable[int] | None, max_k: int) -> list[int]:
    if max_k <= 0:
        if ks is None:
            return [0]
        return [0] if any(int(value) == 0 for value in ks) else []
    if ks is None:
        return list(range(0, max_k + 1))

    normalized = []
    for value in ks:
        k = int(value)
        if k < 0:
            continue
        if k > max_k:
            continue
        normalized.append(k)
    return sorted(set(normalized))


def _prepare_turn_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _validate_episode_frame(frame)
    working = frame.loc[:, list(REQUIRED_EPISODE_COLUMNS)].copy()
    working["episode_id"] = working["episode_id"].astype(str)
    working["task_id"] = working["obs_task_id"].astype(str)
    working["turn"] = working["timestep"].astype(int) + 1
    working["reward"] = working["rewards"].astype(float)
    working["solved_here"] = working["terms"].astype(bool) | (working["reward"] >= 1.0 - 1e-9)
    return working.sort_values(["episode_id", "turn"], ignore_index=True)


def summarize_across_k(
    frame: pd.DataFrame,
    ks: Iterable[int] | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    working = _prepare_turn_frame(frame)

    episodes = (
        working.groupby(["episode_id", "task_id"], as_index=False)
        .agg(num_turns=("turn", "max"))
        .sort_values(["task_id", "episode_id"], ignore_index=True)
    )
    max_k = max(int(episodes["num_turns"].max()) - 1, 0) if not episodes.empty else 0
    k_values = _normalize_ks(ks, max_k)

    overall_rows: list[dict] = []
    per_task_rows: list[dict] = []

    for k in k_values:
        turn_budget = k + 1
        upto_k = working.loc[working["turn"] <= turn_budget, ["episode_id", "task_id", "reward", "solved_here"]]
        per_episode = (
            upto_k.groupby(["episode_id", "task_id"], as_index=False)
            .agg(
                solved_by_k=("solved_here", "max"),
                best_pass_rate_by_k=("reward", "max"),
            )
            .sort_values(["task_id", "episode_id"], ignore_index=True)
        )

        num_episodes = int(len(per_episode))
        num_solved = int(per_episode["solved_by_k"].sum())
        success_rate = float(per_episode["solved_by_k"].mean()) if num_episodes else 0.0
        mean_best_pass_rate = float(per_episode["best_pass_rate_by_k"].mean()) if num_episodes else 0.0

        overall_rows.append(
            {
                "k": k,
                "turn_budget": turn_budget,
                "num_episodes": num_episodes,
                "num_solved": num_solved,
                "success_rate": success_rate,
                "mean_best_pass_rate": mean_best_pass_rate,
            }
        )

        per_task = (
            per_episode.groupby("task_id", as_index=False)
            .agg(
                num_episodes=("episode_id", "nunique"),
                num_solved=("solved_by_k", "sum"),
                success_rate=("solved_by_k", "mean"),
                mean_best_pass_rate=("best_pass_rate_by_k", "mean"),
            )
            .sort_values(["task_id"], ignore_index=True)
        )
        per_task.insert(1, "k", k)
        per_task.insert(2, "turn_budget", turn_budget)
        per_task_rows.extend(per_task.to_dict(orient="records"))

    overall_df = pd.DataFrame(overall_rows)
    per_task_df = pd.DataFrame(per_task_rows)
    summary = {
        "num_episodes": int(episodes["episode_id"].nunique()) if not episodes.empty else 0,
        "num_tasks": int(episodes["task_id"].nunique()) if not episodes.empty else 0,
        "max_k": max_k,
        "k_definition": "number of teacher interactions",
        "k_metrics": overall_rows,
    }
    return summary, overall_df, per_task_df


def save_across_k_artifacts(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    ks: Iterable[int] | None = None,
    stem: str = "code_eval_across_k",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, overall_df, per_task_df = summarize_across_k(frame, ks=ks)

    summary_path = output_dir / f"{stem}_summary.json"
    overall_path = output_dir / f"{stem}_overall.csv"
    per_task_path = output_dir / f"{stem}_per_task.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    overall_df.to_csv(overall_path, index=False)
    per_task_df.to_csv(per_task_path, index=False)

    return {
        "summary_path": str(summary_path),
        "overall_path": str(overall_path),
        "per_task_path": str(per_task_path),
        "summary": summary,
    }
