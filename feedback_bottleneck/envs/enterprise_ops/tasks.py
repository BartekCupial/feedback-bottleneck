from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.enterprise_ops.assets import ensure_seed_db_root

SUPPORTED_DOMAINS = ("calendar", "csm", "drive", "email", "hr", "itsm", "teams", "hybrid")


@dataclass(frozen=True)
class EnterpriseOpsServer:
    gym_name: str
    server_url: str
    seed_database_file: Path
    context: dict[str, Any]


@dataclass(frozen=True)
class EnterpriseOpsTask:
    domain: str
    mode: str
    task_id: str
    system_prompt: str
    user_prompt: str
    selected_tools: tuple[str, ...]
    restricted_tools: tuple[str, ...]
    mcp_endpoint: str
    verifiers: list[dict[str, Any]]
    servers: tuple[EnterpriseOpsServer, ...]


def json_field(row: dict[str, Any], key: str) -> Any:
    value = row[key]
    if isinstance(value, str):
        return json.loads(value)
    return value


def load_enterprise_ops_tasks(config: Args) -> list[EnterpriseOpsTask]:
    env_args = config.enterprise_ops_args
    seed_db_root = Path(env_args.seed_db_root).expanduser()
    domains = tuple(env_args.domains) or SUPPORTED_DOMAINS
    modes = tuple(env_args.modes) or ("oracle",)
    ensure_seed_db_root(seed_db_root, domains=domains)

    tasks: list[EnterpriseOpsTask] = []
    for mode in modes:
        for domain in domains:
            dataset = load_dataset(env_args.dataset_name, mode, split=domain)

            for row in dataset:
                gym_servers_config = json_field(row, "gym_servers_config")
                servers = tuple(
                    EnterpriseOpsServer(
                        gym_name=server_config["mcp_server_name"],
                        server_url=env_args.server_url_overrides.get(server_config["mcp_server_name"])
                        or server_config["mcp_server_url"],
                        seed_database_file=seed_db_root / str(server_config["seed_database_file"]),
                        context=server_config["context"],
                    )
                    for server_config in gym_servers_config
                )
                tasks.append(
                    EnterpriseOpsTask(
                        domain=row["domain"],
                        mode=mode,
                        task_id=row["task_id"],
                        system_prompt=row["system_prompt"],
                        user_prompt=row["user_prompt"],
                        selected_tools=tuple(row["selected_tools"]),
                        restricted_tools=tuple(row["restricted_tools"]),
                        mcp_endpoint=row["mcp_endpoint"],
                        verifiers=json_field(row, "verifiers"),
                        servers=servers,
                    )
                )

    if not tasks:
        raise ValueError("No EnterpriseOps tasks found after applying domain/mode filters.")
    return tasks
