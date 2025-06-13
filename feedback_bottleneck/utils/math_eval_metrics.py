from typing import Any, Dict, Optional

import pandas as pd


def build_cumulative_accuracy_curve_df(args, episode_logs: list[dict[str, Any]]) -> Optional[pd.DataFrame]:
    cumulative_envs = {"math", "countdown", "bbeh", "sciknoweval", "code", "arc_agi", "arc_agi2", "enterprise_ops"}
    if args.env_name not in cumulative_envs:
        return None
    if not episode_logs:
        return None

    required_keys = {"solved", "attempts_used", "max_turns"}
    if not all(required_keys.issubset(log) for log in episode_logs):
        return None

    max_k = max(int(log["max_turns"]) for log in episode_logs)
    if max_k <= 0:
        return None

    problem_to_logs: dict[str, list[dict[str, Any]]] = {}
    for idx, log in enumerate(episode_logs):
        problem_id = str(log["problem_id"]) if "problem_id" in log else f"episode_{idx}"
        problem_to_logs.setdefault(problem_id, []).append(log)

    rows = []
    num_problems = len(problem_to_logs)
    num_episodes = len(episode_logs)
    for k in range(1, max_k + 1):
        problem_scores = []
        for problem_logs in problem_to_logs.values():
            solved_by_k = [
                1.0 if float(log["solved"]) == 1.0 and int(log["attempts_used"]) <= k else 0.0 for log in problem_logs
            ]
            problem_scores.append(sum(solved_by_k) / len(solved_by_k))

        cumulative_accuracy = sum(problem_scores) / len(problem_scores)
        rows.append(
            {
                "K": k,
                "cumulative_accuracy": cumulative_accuracy,
                "num_problems": num_problems,
                "num_episodes": num_episodes,
            }
        )

    return pd.DataFrame(rows)


def build_cumulative_accuracy_metrics(args, episode_logs: list[dict[str, Any]], prefix: str) -> Dict[str, float]:
    curve_df = build_cumulative_accuracy_curve_df(args, episode_logs)
    if curve_df is None or curve_df.empty:
        return {}

    max_k = int(curve_df["K"].max())
    digits = len(str(max_k))
    metrics = {}
    for row in curve_df.itertuples(index=False):
        metrics[f"{prefix}/cumulative_accuracy_at_k/{int(row.K):0{digits}d}"] = float(row.cumulative_accuracy)

    metrics[f"{prefix}/cumulative_accuracy_auc"] = float(curve_df["cumulative_accuracy"].mean())
    return metrics


def build_cumulative_accuracy_wandb_payload(args, episode_logs: list[dict[str, Any]], prefix: str) -> Dict[str, Any]:
    curve_df = build_cumulative_accuracy_curve_df(args, episode_logs)
    if curve_df is None or curve_df.empty:
        return {}

    import wandb

    curve_df["accuracy_first_diff"] = curve_df["cumulative_accuracy"].diff().fillna(curve_df["cumulative_accuracy"])

    first_k_accuracy = curve_df["cumulative_accuracy"].iloc[0]
    curve_df["accuracy_diff_from_first"] = curve_df["cumulative_accuracy"] - first_k_accuracy
    table = wandb.Table(dataframe=curve_df)
    return {
        f"{prefix}/cumacc_table": table,
        f"{prefix}/cumacc_curve": wandb.plot.line(
            table,
            "K",
            "cumulative_accuracy",
            title=f"{prefix} cumulative accuracy vs K",
        ),
        f"{prefix}/gain_step_curve": wandb.plot.line(
            table,
            "K",
            "accuracy_first_diff",
            title=f"{prefix} marginal accuracy gain vs K",
        ),
        f"{prefix}/gain_from_k1_curve": wandb.plot.line(
            table,
            "K",
            "accuracy_diff_from_first",
            title=f"{prefix} accuracy gain from K=1 vs K",
        ),
    }
