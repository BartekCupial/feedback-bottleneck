from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.enterprise_ops.benchmark.mcp_client import MCPClient, create_database_from_file, delete_database
from feedback_bottleneck.envs.enterprise_ops.tasks import EnterpriseOpsServer, EnterpriseOpsTask, load_enterprise_ops_tasks
from feedback_bottleneck.envs.enterprise_ops.verification import run_enterprise_ops_verifiers
from feedback_bottleneck.envs.env_wrapper import EnvWrapper


@dataclass
class EnterpriseOpsRuntime:
    server: EnterpriseOpsServer
    database_id: str
    client: MCPClient
    tools: list[dict[str, Any]]


class IdentityLanguageActionSpace:
    def __init__(self, max_action_length: int):
        self.max_action_length = max_action_length

    def __contains__(self, action: str) -> bool:
        return isinstance(action, str)

    def map(self, action: str) -> str:
        text = "" if action is None else str(action)
        return text[: self.max_action_length].strip()


def _run_awaitable(awaitable):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}

    def run_in_thread():
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result["value"]


def build_enterprise_ops_tool_route(
    task: EnterpriseOpsTask,
    tools_by_gym: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    selected_tools = task.selected_tools
    restricted_tools = set(task.restricted_tools)
    tool_choices: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for gym_name, tools in tools_by_gym.items():
        for tool in tools:
            tool_name = str(tool.get("name", ""))
            if not tool_name or tool_name in restricted_tools:
                continue
            tool_choices.setdefault(tool_name, []).append((gym_name, tool))

    if selected_tools:
        missing_tools = [name for name in selected_tools if name not in tool_choices]
        if missing_tools:
            raise ValueError(
                f"EnterpriseOps MCP server did not expose selected tools for task {task.task_id}: {missing_tools}"
            )
        selected_names = selected_tools
    else:
        selected_names = tuple(sorted(tool_choices))

    ambiguous_tools = {
        name: [gym_name for gym_name, _ in tool_choices[name]] for name in selected_names if len(tool_choices[name]) > 1
    }
    if ambiguous_tools:
        raise ValueError(
            f"EnterpriseOps selected tools are ambiguous across MCP servers for task {task.task_id}: "
            f"{ambiguous_tools}"
        )

    tool_to_gym = {}
    selected = []
    for name in selected_names:
        gym_name, tool = tool_choices[name][0]
        tool_with_gym = dict(tool)
        tool_with_gym["gym_name"] = gym_name
        selected.append(tool_with_gym)
        tool_to_gym[name] = gym_name
    return selected, tool_to_gym


class EnterpriseOpsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_name: str,
        task: str,
        config: Args,
        render_mode: Optional[str] = None,
        llm_judge=None,
    ):
        super().__init__()
        self.env_name = env_name
        self.task = task
        self.config = config
        self.max_steps = int(config.enterprise_ops_args.max_turns)
        self.max_action_length = int(config.enterprise_ops_args.max_action_length)
        self.language_action_space = IdentityLanguageActionSpace(self.max_action_length)
        self.default_action = ""
        self.action_space = gym.spaces.Text(max_length=self.max_action_length)
        self.observation_space = gym.spaces.Dict({})
        self.actions = []

        self.dataset = load_enterprise_ops_tasks(config)
        self._episode_index = 0
        self._current_task: Optional[EnterpriseOpsTask] = None
        self._runtimes: dict[str, EnterpriseOpsRuntime] = {}
        self._tool_to_gym: dict[str, str] = {}
        self._tools: list[dict[str, Any]] = []
        self._attempts: list[dict[str, Any]] = []
        self._final_response = ""
        self._verification_summary: dict[str, Any] = {}

    def get_wrapper_attr(self, name: str):
        return getattr(self, name)

    def get_text_action(self, action):
        return action

    def check_action_validity(self, candidate_action):
        return self.language_action_space.map(candidate_action), None

    def get_stats(self):
        tool_calls = sum(1 for attempt in self._attempts if attempt.get("kind") == "tool_call")
        invalid_actions = sum(1 for attempt in self._attempts if attempt.get("kind") == "invalid")
        stats = {
            "max_turns": self.max_steps,
            "num_attempts": len(self._attempts),
            "num_tools": len(self._tools),
            "tool_calls": tool_calls,
            "invalid_actions": invalid_actions,
            "domain": self._current_task.domain if self._current_task else "",
            "mode": self._current_task.mode if self._current_task else "",
            "dataset_task_id": self._current_task.task_id if self._current_task else "",
            "num_servers": len(self._current_task.servers) if self._current_task else 0,
            "database_ids": self._database_ids_by_gym(),
        }
        if self._verification_summary:
            stats.update(self._verification_summary)
        return stats

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._cleanup_database()
        self._attempts = []
        self._tools = []
        self._tool_to_gym = {}
        self._final_response = ""
        self._verification_summary = {}

        if seed is None:
            idx = self._episode_index % len(self.dataset)
            self._episode_index += 1
        else:
            idx = int(seed) % len(self.dataset)

        self._current_task = self.dataset[idx]
        runtimes: dict[str, EnterpriseOpsRuntime] = {}
        try:
            for server in self._current_task.servers:
                database_id = create_database_from_file(server.server_url, str(server.seed_database_file))
                if not database_id:
                    raise RuntimeError(f"Failed to create EnterpriseOps database from {server.seed_database_file}")

                try:
                    client = MCPClient(
                        base_url=server.server_url,
                        database_id=database_id,
                        mcp_endpoint=self._current_task.mcp_endpoint,
                        context=server.context,
                    )
                    connected = _run_awaitable(client.connect())
                    if not connected:
                        raise RuntimeError(f"Failed to connect to EnterpriseOps MCP server: {server.server_url}")

                    tools = _run_awaitable(client.list_tools())
                    runtimes[server.gym_name] = EnterpriseOpsRuntime(
                        server=server,
                        database_id=database_id,
                        client=client,
                        tools=tools,
                    )
                except Exception:
                    delete_database(server.server_url, database_id)
                    raise

            tools_by_gym = {gym_name: runtime.tools for gym_name, runtime in runtimes.items()}
            tools, tool_to_gym = build_enterprise_ops_tool_route(self._current_task, tools_by_gym)
            if not tools:
                raise RuntimeError(f"EnterpriseOps MCP servers exposed no tools for task {self._current_task.task_id}")
        except Exception:
            self._cleanup_runtimes(runtimes)
            raise

        self._runtimes = runtimes
        self._tools = tools
        self._tool_to_gym = tool_to_gym

        info = {
            "correct": False,
            "problem_id": self._current_task.task_id,
            "source": self._source_summary(),
            "domain": self._current_task.domain,
            "mode": self._current_task.mode,
            "dataset_task_id": self._current_task.task_id,
            "database_ids": self._database_ids_by_gym(),
        }
        return self._build_observation(), info

    def step(self, action: str):
        if self._current_task is None or not self._runtimes:
            raise RuntimeError("Environment must be reset before stepping.")

        action_text = self.language_action_space.map(action)
        attempt_no = len(self._attempts) + 1
        parsed = self._parse_action(action_text)

        if parsed["kind"] == "invalid":
            feedback = parsed["error"]
            self._attempts.append(
                {
                    "turn": attempt_no,
                    "action": action_text,
                    "kind": "invalid",
                    "feedback": feedback,
                }
            )
            return self._transition(reward=0.0, terminated=False, feedback=feedback)

        if parsed["kind"] == "final":
            self._final_response = parsed["final_response"]
            feedback = "Final response recorded. Verification will run now."
            self._attempts.append(
                {
                    "turn": attempt_no,
                    "action": action_text,
                    "kind": "final",
                    "final_response": self._final_response,
                    "feedback": feedback,
                }
            )
            return self._transition(reward=0.0, terminated=True, feedback=feedback)

        gym_name = self._tool_to_gym.get(parsed["tool_name"])
        if gym_name is None:
            feedback = f"EnterpriseOps tool is not available for this task: {parsed['tool_name']}"
            self._attempts.append(
                {
                    "turn": attempt_no,
                    "action": action_text,
                    "kind": "invalid",
                    "feedback": feedback,
                }
            )
            return self._transition(reward=0.0, terminated=False, feedback=feedback)

        runtime = self._runtimes[gym_name]
        tool_result = _run_awaitable(
            runtime.client.call_tool(
                parsed["tool_name"],
                parsed["arguments"],
                database_id=runtime.database_id,
            )
        )
        feedback = self._format_tool_feedback(tool_result)
        self._attempts.append(
            {
                "turn": attempt_no,
                "action": action_text,
                "kind": "tool_call",
                "gym_name": gym_name,
                "tool_name": parsed["tool_name"],
                "arguments": parsed["arguments"],
                "tool_result": tool_result,
                "feedback": feedback,
            }
        )
        return self._transition(reward=0.0, terminated=False, feedback=feedback)

    async def apply_post_step_verification(
        self,
        action: str,
        obs,
        reward,
        terminated,
        truncated,
        info,
    ):
        if not (terminated or truncated):
            return obs, reward, terminated, truncated, info

        if not self._current_task.verifiers:
            raise ValueError(f"EnterpriseOps task {self._current_task.task_id} has no verifiers.")

        if not self._runtimes:
            raise RuntimeError("EnterpriseOps verification requires an active MCP client and database.")

        verified_reward, feedback, verification_results, self._verification_summary = (
            await run_enterprise_ops_verifiers(
                self._current_task,
                self._mcp_clients_by_gym(),
                self._database_ids_by_gym(),
                self._final_response,
                self._attempts,
            )
        )
        self._attempts[-1]["feedback"] = feedback

        stats = dict(info.get("episode_extra_stats", {}))
        stats.update({"correct": verified_reward, "solved": verified_reward})
        stats.update(self._verification_summary)
        info = dict(info)
        info.update(
            {
                "correct": bool(verified_reward),
                "last_feedback": feedback,
                "verification_results": verification_results,
                "verification_summary": self._verification_summary,
                "episode_extra_stats": stats,
            }
        )
        return self._build_observation(), verified_reward, terminated, truncated, info

    def close(self):
        self._cleanup_database()
        super().close()

    def _cleanup_database(self):
        self._cleanup_runtimes(self._runtimes)
        self._runtimes = {}

    def _cleanup_runtimes(self, runtimes: dict[str, EnterpriseOpsRuntime]):
        for runtime in runtimes.values():
            delete_database(runtime.server.server_url, runtime.database_id)

    def _database_ids_by_gym(self) -> dict[str, str]:
        return {gym_name: runtime.database_id for gym_name, runtime in self._runtimes.items()}

    def _mcp_clients_by_gym(self) -> dict[str, MCPClient]:
        return {gym_name: runtime.client for gym_name, runtime in self._runtimes.items()}

    def _source_summary(self) -> dict[str, str]:
        return {server.gym_name: str(server.seed_database_file) for server in self._current_task.servers}

    def _transition(self, *, reward: float, terminated: bool, feedback: str):
        truncated = (not terminated) and len(self._attempts) >= self.max_steps
        remaining_turns = max(self.max_steps - len(self._attempts), 0)
        info = {
            "correct": bool(reward == 1.0),
            "problem_id": self._current_task.task_id,
            "source": self._source_summary(),
            "domain": self._current_task.domain,
            "mode": self._current_task.mode,
            "dataset_task_id": self._current_task.task_id,
            "database_ids": self._database_ids_by_gym(),
            "attempts_used": len(self._attempts),
            "remaining_turns": remaining_turns,
            "last_feedback": feedback,
            "episode_extra_stats": {
                "correct": float(reward == 1.0),
                "solved": float(reward == 1.0),
                "attempts_used": len(self._attempts),
                "remaining_turns": remaining_turns,
                "tool_calls": sum(1 for attempt in self._attempts if attempt.get("kind") == "tool_call"),
                "invalid_actions": sum(1 for attempt in self._attempts if attempt.get("kind") == "invalid"),
            },
        }
        if terminated:
            info["end_status"] = "final_response"
        elif truncated:
            info["end_status"] = "max_turns"

        return self._build_observation(), reward, terminated, truncated, info

    def _build_observation(self) -> dict[str, Any]:
        task = self._current_task
        last_feedback = self._attempts[-1]["feedback"] if self._attempts else "No tool calls yet."

        return {
            "obs": {
                "problem": task.user_prompt,
                "problem_id": task.task_id,
                "system_prompt": task.system_prompt,
                "servers": [server.gym_name for server in task.servers],
                "tools": self._tools,
                "last_feedback": last_feedback,
            },
            "text": {
                "short_term_context": last_feedback,
                "long_term_context": task.user_prompt,
            },
        }

    def _parse_action(self, action_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(action_text)
        except json.JSONDecodeError as exc:
            return {
                "kind": "invalid",
                "error": f"Invalid EnterpriseOps action JSON: {exc}",
            }

        if not isinstance(payload, dict):
            return {"kind": "invalid", "error": "EnterpriseOps action must be a JSON object."}

        if "final_response" in payload:
            return {
                "kind": "final",
                "final_response": str(payload["final_response"]),
            }

        tool_name = payload.get("tool_name") or payload.get("name") or payload.get("tool")
        if not tool_name:
            return {
                "kind": "invalid",
                "error": "EnterpriseOps tool action requires `tool_name`, `name`, or `tool`.",
            }

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            return {"kind": "invalid", "error": "EnterpriseOps tool `arguments` must be a JSON object."}

        return {
            "kind": "tool_call",
            "tool_name": str(tool_name),
            "arguments": arguments,
        }

    def _format_tool_feedback(self, tool_result: dict[str, Any]) -> str:
        if not tool_result.get("success"):
            return f"Tool call failed: {tool_result.get('error')}"
        if tool_result.get("error"):
            return f"Tool returned an error: {tool_result['error']}"
        return json.dumps(tool_result.get("result"), ensure_ascii=True, default=str)


def make_enterprise_ops_env(env_name, task, config, render_mode: Optional[str] = None, llm_judge=None):
    env = EnterpriseOpsEnv(env_name, task, config, render_mode=render_mode, llm_judge=llm_judge)
    return EnvWrapper(env, env_name, task, args=config)
