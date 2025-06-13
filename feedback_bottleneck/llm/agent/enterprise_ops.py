import json
import re
from collections import deque
from typing import Any, Dict

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.llm.agent.base import BaseAgent, BaseFormatter

JSON_OBS_KEYS = {
    "tools",
}


def _text_message(role: str, text: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


def _truncate(text: str, max_chars: int) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _compact_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for tool in tools:
        compact.append(
            {
                "gym_name": tool.get("gym_name", ""),
                "name": tool["name"],
                "description": _truncate(tool.get("description", ""), 700),
                "input_schema": tool.get("inputSchema", {}),
            }
        )
    return compact


ACTION_INSTRUCTIONS = (
    "Choose exactly one next action.\n"
    "To call a tool, return:\n"
    '{"tool_name": "tool_name_here", "arguments": {"arg": "value"}}\n'
    "When the task is complete, return:\n"
    '{"final_response": "brief summary"}\n'
    "Do not wrap the JSON in markdown."
)


def _trim_history_for_user_turn(history: deque, max_turns: int) -> None:
    assert max_turns >= 0
    max_messages = 1 + 2 * max_turns
    if len(history) <= max_messages:
        return

    first_user = history[0]
    recent_messages = list(history)[-(max_messages - 1) :] if max_messages > 1 else []
    history.clear()
    history.append(first_user)
    history.extend(recent_messages)


def _render_initial_user_turn(obs: dict[str, Any]) -> str:
    inner = obs["obs"]
    tools = _compact_tools(inner["tools"])

    return "\n\n".join(
        [
            obs["text"]["long_term_context"],
            f"Available tools:\n{json.dumps(tools, indent=2, ensure_ascii=True, default=str)}",
            ACTION_INSTRUCTIONS,
        ]
    )


def _render_result_turn(obs: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"Tool result:\n{obs['text']['short_term_context']}",
            ACTION_INSTRUCTIONS,
        ]
    )


def _extract_json_action(text: str) -> str:
    text = str(text).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1].strip()
    return text


def _build_observation_from_step(step_data: Dict[str, Any]) -> Dict[str, Any]:
    inner = {}
    for key, value in step_data.items():
        if not key.startswith("obs_"):
            continue

        obs_key = key.removeprefix("obs_")
        if obs_key in JSON_OBS_KEYS and isinstance(value, str) and value[:1] in "[{":
            value = json.loads(value)
        inner[obs_key] = value

    return {
        "obs": inner,
        "text": {
            "short_term_context": step_data["short_term_context"],
            "long_term_context": step_data["long_term_context"],
        },
    }


class NaiveEnterpriseOpsFormatter(BaseFormatter):
    def __init__(self, args: Args):
        super().__init__(args)
        self.current_obs = None

    def reset(self):
        super().reset()
        self.current_obs = None
        self.chat_history = deque()

    def append_observation(self, obs: Dict[str, Any]):
        super().append_observation(obs)
        self.current_obs = obs

    def append_user_turn(self, obs: Dict[str, Any]):
        if self.act_history:
            prompt_text = _render_result_turn(obs)
        else:
            prompt_text = _render_initial_user_turn(obs)

        self.chat_history.append(_text_message("user", prompt_text))
        _trim_history_for_user_turn(self.chat_history, self.max_history)

    def append_assistant_turn(self, action: str):
        self.chat_history.append(_text_message("assistant", action))

    def get_prompt(self, role: str = "actor", **kwargs):
        del role, kwargs
        return [_text_message("system", self.current_obs["obs"]["system_prompt"]), *self.chat_history]

    def generate_sft_samples(self, episode_iterator):
        self.reset()
        for step_data in episode_iterator:
            observation = _build_observation_from_step(step_data)
            self.append_observation(observation)
            self.append_user_turn(observation)

            target = step_data["text_actions"]
            yield self.get_prompt(role="actor"), target

            self.append_action(target)
            self.append_assistant_turn(target)


class NaiveEnterpriseOpsAgent(BaseAgent):
    async def get_action(self, obs: Dict[str, Any]):
        self.prompt_formatter.append_observation(obs)
        self.prompt_formatter.append_user_turn(obs)
        messages = self.prompt_formatter.get_prompt(role="actor")
        output = await self.generate(messages)
        raw_output = output["text"].strip()
        action = _extract_json_action(raw_output)
        self.prompt_formatter.append_action(action)
        self.prompt_formatter.append_assistant_turn(action)

        return {
            "action": action,
            "plan": "",
            "judge_feedback": "",
            "logprob": output["logprobs"],
            "prompt_token_ids": output["prompt_token_ids"],
            "action_token_ids": output["token_ids"],
            "raw_output": raw_output,
        }


class HierarchicalEnterpriseOpsFormatter(NaiveEnterpriseOpsFormatter):
    def __init__(self, args: Args):
        raise ValueError("EnterpriseOps uses llm_agent='naive'; teacher feedback is not part of this environment.")


class HierarchicalEnterpriseOpsAgent(NaiveEnterpriseOpsAgent):
    pass


class HierarchicalSeparateEnterpriseOpsAgent(NaiveEnterpriseOpsAgent):
    pass
