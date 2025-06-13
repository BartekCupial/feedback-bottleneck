import math
import re
import xml.dom.minidom
from collections import deque
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import ray

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.nle.actions import MINIHACK_ACTIONS, NLE_ACTIONS, NLE_DESCRIPTIONS
from feedback_bottleneck.envs.nle.instruction_prompt import format_action_list, get_task_goal
from feedback_bottleneck.envs.nle.tips import GAME_MECHANICS_TIPS, WIKI_TIPS
from feedback_bottleneck.llm.agent.base import BaseAgent, BaseFormatter

COMMANDS_LIST = format_action_list(NLE_ACTIONS, [NLE_DESCRIPTIONS[action] for action in NLE_ACTIONS])


class XMLBuilder:
    @staticmethod
    def tag(name: str, content: Any) -> str:
        content_str = str(content).strip()
        return f"<{name}>\n{content_str}\n</{name}>"


def ascii_render(chars):
    rows, cols = chars.shape
    result = []
    for i in range(rows):
        result_row = ""
        for j in range(cols):
            entry = "<" + chr(chars[i, j]) + ">"
            result_row += entry
        result.append(result_row)
    return "\n".join(result)


def map_render(chars) -> str:
    ascii_map = ascii_render(chars)
    lines = ascii_map.split("\n")
    ascii_map = "\n".join([""] + lines[1:-2])

    return ascii_map


def build_judge_system_prompt(task_goal: str) -> str:
    persona = f"You are an expert NetHack player. Your task is to guide a beginner player by providing subgoals to achieve: {task_goal}"

    instructions = [
        "Analyze the `current_state` and `current_subgoal`.",
        "Decide: KEEP (goal is valid/achievable) or UPDATE (goal is finished, invalid, or dangerous).",
        "If UPDATE, you MUST provide a specific new subgoal.",
        "The subgoal should be clear and achievable within 10-20 steps.",
        "The subgoal should be short and concise.",
    ]

    rules = XMLBuilder.tag("task_instruction", "\n".join(f"- {i}" for i in instructions))
    tips = XMLBuilder.tag("tips", f"{WIKI_TIPS}\n{GAME_MECHANICS_TIPS}")

    fmt = XMLBuilder.tag(
        "output_format",
        XMLBuilder.tag("status", "KEEP or UPDATE") + "\n" + XMLBuilder.tag("goal", "New subgoal text (if UPDATE)"),
    )

    return f"{persona}\n\n{rules}\n\n{tips}\n\n{fmt}"


def build_beginner_naive_actor_system_prompt(task_goal: str) -> str:
    persona = f"You are a beginner NetHack player. Your task is to follow expert subgoals to achieve: {task_goal}"

    instructions = [
        "Analyze the `current_state` and the `current_subgoal`.",
        "Choose the best immediate action from the command list.",
        "Provide ONLY the action command inside <action> tags.",
    ]

    rules = XMLBuilder.tag("task_instruction", "\n".join(f"- {i}" for i in instructions))
    tips = XMLBuilder.tag("tips", GAME_MECHANICS_TIPS)
    cmds = XMLBuilder.tag("commands", COMMANDS_LIST)
    fmt = XMLBuilder.tag(
        "output_format",
        XMLBuilder.tag("action", "(The immediate command to execute, e.g., 'north', 'fight', 'search')"),
    )

    return f"{persona}\n\n{rules}\n\n{tips}\n\n{cmds}\n\n{fmt}"


def build_beginner_naive_actor_tail_prompt(task_goal: str, include_subgoal: bool) -> str:
    instructions = [
        "Choose the best immediate action from the command list.",
        "If <current_subgoal> exists, prioritize actions that advance it.",
        "Output ONLY <action>...</action>.",
    ]
    if not include_subgoal:
        instructions[1] = "Focus on immediate survival and progress toward the task goal."

    prompt = [
        XMLBuilder.tag("objective", task_goal),
        XMLBuilder.tag("task_instruction", "\n".join(f"- {i}" for i in instructions)),
        XMLBuilder.tag("commands", COMMANDS_LIST),
        XMLBuilder.tag(
            "output_format",
            XMLBuilder.tag("action", "(The immediate action to execute, e.g., 'north', 'fight', 'search')"),
        ),
    ]
    return "\n\n".join(prompt)


def build_actor_messages(
    *,
    args: Args,
    user_text: str,
    system_text: str,
    include_subgoal: bool,
    include_system_prompt: bool = True,
) -> List[Dict[str, str]]:
    layout = args.actor_prompt_layout
    tail_style = args.actor_prompt_tail_style

    if include_system_prompt:
        if layout == "prompt_last":
            tail_text = (
                system_text
                if tail_style == "full"
                else build_beginner_naive_actor_tail_prompt(get_task_goal(args.task), include_subgoal=include_subgoal)
            )
            merged_user = "\n\n".join([user_text, XMLBuilder.tag("policy_prompt", tail_text)])
            return [{"role": "user", "content": [{"type": "text", "text": merged_user}]}]

        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

    else:
        return [
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]


def build_expert_naive_actor_system_prompt(task_goal: str) -> str:
    persona = f"You are an expert NetHack player. Your task is to: {task_goal}"

    instructions = [
        "Analyze the `current_state`",
        "Choose the best immediate action from the command list.",
        "Provide ONLY the action command inside <action> tags.",
    ]

    prompt = [
        persona,
        XMLBuilder.tag("task_instruction", "\n".join(f"- {i}" for i in instructions)),
        XMLBuilder.tag("tips", f"{WIKI_TIPS}\n{GAME_MECHANICS_TIPS}"),
        XMLBuilder.tag("commands", COMMANDS_LIST),
        XMLBuilder.tag(
            "output_format",
            XMLBuilder.tag("action", "(The immediate action to execute, e.g., 'north', 'fight', 'search')"),
        ),
    ]

    return "\n\n".join(prompt)


def build_plan_completion_prompt(current_goal, plan_observation_history, plan_action_history) -> str:
    chat_template = []

    # add system prompt
    sys_text = """Based on the plan and execution history, has the plan been successfully completed?
Think about it, then respond EXACTLY with either: <eval>YES</eval> or <eval>NO</eval>"""
    chat_template.append({"role": "system", "content": [{"type": "text", "text": sys_text}]})

    # add current goal
    goal_text = XMLBuilder.tag("current_goal", current_goal)
    chat_template.append({"role": "user", "content": [{"type": "text", "text": goal_text}]})

    # add interaction history
    for obs, act in zip(plan_observation_history[:15], plan_action_history[:15]):
        obs_block = build_observation_block([obs], include_map=True)
        act_block = XMLBuilder.tag("action", act)
        chat_template.append({"role": "user", "content": [{"type": "text", "text": obs_block}]})
        chat_template.append({"role": "assistant", "content": [{"type": "text", "text": act_block}]})

    return chat_template


def build_partial_plan_completion_prompt(current_goal, plan_observation_history, plan_action_history) -> str:
    chat_template = []

    # add system prompt
    sys_text = """Based on the plan and execution history, estimate the progress toward the goal.
Respond with a score between 0.0 and 1.0 inside <progress> tags.

Rubric:
0.0: No progress made, or the agent failed/died.
0.25: The agent started the task (e.g., moved towards target) but is far from completion.
0.5: Significant progress (e.g., halfway there, or acquired necessary intermediate items).
0.75: Very close to completion, only one or two steps missing.
1.0: The plan is fully and successfully completed.

Output EXACTLY like this example: <progress>0.5</progress>"""
    chat_template.append({"role": "system", "content": [{"type": "text", "text": sys_text}]})

    # add current goal
    goal_text = XMLBuilder.tag("current_goal", current_goal)
    chat_template.append({"role": "user", "content": [{"type": "text", "text": goal_text}]})

    # add interaction history
    for obs, act in zip(plan_observation_history[:15], plan_action_history[:15]):
        obs_block = build_observation_block([obs], include_map=True)
        act_block = XMLBuilder.tag("action", act)
        chat_template.append({"role": "user", "content": [{"type": "text", "text": obs_block}]})
        chat_template.append({"role": "assistant", "content": [{"type": "text", "text": act_block}]})

    return chat_template


def build_plan_adherence_prompt(current_goal, plan_observation_history, plan_action_history) -> str:
    """
    Evaluates if the agent adhered to the plan, even if the plan was interrupted or partially completed.
    """
    chat_template = []

    # add system prompt
    sys_text = """You are evaluating "Plan Adherence" for a NetHack agent.
Analyze the execution history relative to the `current_goal`.
Note: The goal is being updated now, possibly because it was completed OR because the situation changed.

Respond EXACTLY with <eval>YES</eval> if:
1. The agent successfully completed the goal.
2. The agent was actively working towards the goal and made valid progress (even if not finished).
3. The agent followed the instructions faithfully until the new goal was issued.

Respond EXACTLY with <eval>NO</eval> if:
1. The agent completely ignored the goal.
2. The agent made no progress or wasted time on unrelated actions without justification.
3. The agent died or failed the task completely due to negligence."""

    chat_template.append({"role": "system", "content": [{"type": "text", "text": sys_text}]})

    # add current goal
    goal_text = XMLBuilder.tag("current_goal", current_goal)
    chat_template.append({"role": "user", "content": [{"type": "text", "text": goal_text}]})

    # add interaction history
    for obs, act in zip(plan_observation_history[:15], plan_action_history[:15]):
        obs_block = build_observation_block([obs], include_map=True)
        act_block = XMLBuilder.tag("action", act)
        chat_template.append({"role": "user", "content": [{"type": "text", "text": obs_block}]})
        chat_template.append({"role": "assistant", "content": [{"type": "text", "text": act_block}]})

    return chat_template


def build_adaptive_replan_system_prompt(new_goal, plan_observation_history, plan_action_history) -> str:
    chat_template = []

    # add system prompt
    sys_text = """You are verifying whether proposing a new plan was justified. Analyze the interaction
history up to the current plan proposal. Respond YES if either: 1. The previous
plan was completed before this plan started, or 2. The new plan is clearly motivated
by unexpected circumstances (new threat, low health/hunger/energy, etc.). Otherwise
respond NO. Respond EXACTLY with <eval>YES</eval> or <eval>NO</eval>."""
    chat_template.append({"role": "system", "content": [{"type": "text", "text": sys_text}]})

    # add interaction history
    for obs, act in zip(plan_observation_history[:15], plan_action_history[:15]):
        obs_block = build_observation_block([obs], include_map=True)
        act_block = XMLBuilder.tag("action", act)
        chat_template.append({"role": "user", "content": [{"type": "text", "text": obs_block}]})
        chat_template.append({"role": "assistant", "content": [{"type": "text", "text": act_block}]})

    # add new goal
    goal_text = XMLBuilder.tag("new_goal", new_goal)
    chat_template.append({"role": "user", "content": [{"type": "text", "text": goal_text}]})

    return chat_template


def build_observation_block(obs_history: List[Dict[str, Any]], include_map: bool) -> str:
    last_obs = obs_history[-1]["obs"]

    # Message log from history
    messages = [obs["obs"]["text_message"] for obs in obs_history]

    components = [
        XMLBuilder.tag("message_log", "\n".join(messages)),
        XMLBuilder.tag("cursor", last_obs["text_cursor"]),
        XMLBuilder.tag("overview", last_obs["text_overview"]),
        XMLBuilder.tag("map description", last_obs["text_map"]),
        XMLBuilder.tag("stats", last_obs["text_blstats"]),
        XMLBuilder.tag("language_observation", last_obs["text_glyphs"]),
        XMLBuilder.tag("prayer_status", last_obs["text_prayer"]),
        XMLBuilder.tag("inventory", last_obs["text_inventory"]),
    ]

    if include_map:
        map_ascii = map_render(last_obs["tty_chars"])
        components.insert(4, XMLBuilder.tag("map", map_ascii))

    return XMLBuilder.tag("current_state", "\n".join(components))


def build_actions_history_block(act_history: List[str]) -> str:
    return XMLBuilder.tag("action_history", "\n".join(act_history))


def build_subgoal_block(current_goal: str, goal_age: int) -> str:
    content = [current_goal, XMLBuilder.tag("goal_age", goal_age)]

    if goal_age > 20:
        content.append(XMLBuilder.tag("warning", "Subgoal is stagnating. Consider if it is still viable."))

    return XMLBuilder.tag("current_subgoal", "\n".join(map(str, content)))


def build_nle_obs(step_data) -> Dict[str, Any]:
    nle_obs = {}
    for key, val in step_data.items():
        if key.startswith("obs_"):
            clean_key = key[4:]  # Remove 'obs_' prefix
            if isinstance(val, list):
                nle_obs[clean_key] = np.asanyarray(val)
            else:
                nle_obs[clean_key] = val

    return {
        "obs": nle_obs,
        "short_term_context": step_data["short_term_context"],
        "long_term_context": step_data["long_term_context"],
    }


class NaiveNLEFormatter(BaseFormatter):
    def get_prompt(self, role: str = "actor", **kwargs) -> List[Dict[str, str]]:
        task_goal = get_task_goal(self.args.task)

        sys_text = build_expert_naive_actor_system_prompt(task_goal)

        user_text = "\n\n".join(
            [
                XMLBuilder.tag("action_history", "\n".join(self.act_history)),
                build_observation_block(self.obs_history, include_map=True),
            ]
        )
        return build_actor_messages(
            args=self.args,
            user_text=user_text,
            system_text=sys_text,
            include_subgoal=False,
        )

    def generate_sft_samples(self, episode_iterator):
        self.reset()

        for step_data in episode_iterator:
            observation = build_nle_obs(step_data)
            self.append_observation(observation)

            messages = self.get_prompt(role="actor")
            action_text = step_data["valid_text_actions"]
            target = XMLBuilder.tag("action", action_text)
            yield messages, target

            self.append_action(action_text)


class HierarchicalNLEFormatter(BaseFormatter):
    def get_prompt(self, role: str, current_goal: str, goal_age: int) -> List[Dict[str, str]]:
        task_goal = get_task_goal(self.args.task)

        if role == "judge":
            sys_text = build_judge_system_prompt(task_goal)
        else:
            sys_text = build_beginner_naive_actor_system_prompt(task_goal)

        user_text = "\n\n".join(
            [
                XMLBuilder.tag("action_history", "\n".join(self.act_history)),
                build_observation_block(self.obs_history, include_map=True),
                XMLBuilder.tag("current_subgoal", current_goal),
            ]
        )

        if role == "judge":
            return [
                {"role": "system", "content": [{"type": "text", "text": sys_text}]},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ]

        return build_actor_messages(
            args=self.args,
            user_text=user_text,
            system_text=sys_text,
            include_subgoal=True,
        )

    def generate_sft_samples(self, episode_iterator):
        self.reset()
        current_goal = "[NO GOAL ASSIGNED]"
        goal_age = 0

        for step_data in episode_iterator:
            new_goal = step_data["plans"]

            if new_goal != current_goal:
                current_goal = new_goal
                goal_age = 0
            else:
                goal_age += 1

            observation = build_nle_obs(step_data)
            self.append_observation(observation)

            messages = self.get_prompt(
                role="actor",
                current_goal=current_goal,
                goal_age=goal_age,
            )
            action_text = step_data["valid_text_actions"]
            target = XMLBuilder.tag("action", action_text)
            yield messages, target

            self.append_action(action_text)

    def find_goal_timesteps(self, episode_steps) -> List[int]:
        """
        Pre-process the episode to find the timesteps at which the goal changes.
        This allows us to efficiently generate TRA samples later.
        """
        H = len(episode_steps)

        current_goal = ""
        next_goal_index = H - 1  # -1 so we dont go OOB
        next_plan_timesteps = []
        for i in reversed(range(H)):
            step_data = episode_steps[i]
            new_goal = step_data["plans"]

            if current_goal != new_goal:
                current_goal = new_goal
                next_goal_index = i + 1
            next_plan_timesteps.append(next_goal_index)
        next_plan_timesteps.reverse()

        return next_plan_timesteps

    def sample_geometric(self, gamma: float, remaining_steps: int) -> int:
        """Sample k from a geometric distribution truncated at remaining_steps."""

        if remaining_steps <= 1:
            return remaining_steps

        max_cdf = 1.0 - (gamma**remaining_steps)
        u = np.random.uniform(0, max_cdf)
        k = math.ceil(math.log(1 - u) / math.log(gamma))

        return int(k)

    def generate_tra_samples(self, episode_iterator, gamma: float = 0.9):
        self.reset()
        current_goal = "[NO GOAL ASSIGNED]"
        goal_age = 0

        episode_steps = list(episode_iterator)
        H = len(episode_steps)

        goal_timesteps = self.find_goal_timesteps(episode_steps)

        for i in range(H):
            step_data = episode_steps[i]
            new_goal = step_data["plans"]

            if new_goal != current_goal:
                current_goal = new_goal
                goal_age = 0
            else:
                goal_age += 1

            observation = build_nle_obs(step_data)
            self.append_observation(observation)

            messages = self.get_prompt(
                role="actor",
                current_goal=current_goal,
                goal_age=goal_age,
            )
            action_text = step_data["valid_text_actions"]
            target = XMLBuilder.tag("action", action_text)

            remaining_steps = goal_timesteps[i] - i
            k = self.sample_geometric(gamma, remaining_steps)
            future_index = min(i + k, H - 1)

            state_t_content = build_observation_block(
                [{"obs": observation["obs"]}], include_map=self.args.tra_include_map
            )

            future_observation = build_nle_obs(episode_steps[future_index])
            state_future_content = build_observation_block(
                [{"obs": future_observation["obs"]}], include_map=self.args.tra_include_map
            )

            instruction_content = XMLBuilder.tag("current_subgoal", current_goal)

            task_goal = get_task_goal(self.args.task)
            sys_text = build_beginner_naive_actor_system_prompt(task_goal)

            state_instruction_content = "\n\n".join([state_t_content, instruction_content])
            state_instruction = build_actor_messages(
                args=self.args,
                user_text=state_instruction_content,
                system_text=sys_text,
                include_subgoal=False,
                include_system_prompt=self.args.tra_include_system_prompt,
            )

            state_future_state_content = "\n\n".join([state_future_content])
            state_future_state = build_actor_messages(
                args=self.args,
                user_text=state_future_state_content,
                system_text=sys_text,
                include_subgoal=False,
                include_system_prompt=self.args.tra_include_system_prompt,
            )

            yield messages, target, state_instruction, state_future_state
            self.append_action(action_text)


class NaiveNLEAgent(BaseAgent):
    async def get_action(self, obs):
        self.prompt_formatter.append_observation(obs)

        messages = self.prompt_formatter.get_prompt(role="actor")
        output = await self.generate(messages)
        action = self.parse_tag_content(output["text"], ["action"])[0] or "noop"

        self.prompt_formatter.append_action(action)

        return dict(
            action=action,
            plan="",
            logprob=output["logprobs"],
            prompt_token_ids=output["prompt_token_ids"],
            action_token_ids=output["token_ids"],
            raw_output=output["text"],
        )


class HierarchicalNLEAgent(BaseAgent):
    def __init__(self, llm_actor, prompt_formatter, args: Args):
        super().__init__(llm_actor, prompt_formatter, args)
        self.ask_for_plan_completion = args.ask_for_plan_completion
        self.ask_for_adaptive_replan = args.ask_for_adaptive_replan

    def reset(self):
        super().reset()
        self.current_goal = "[NO GOAL ASSIGNED]"
        self.goal_age = 0

        self.plan_action_history = []
        self.plan_observation_history = []
        self.previous_plan = None

    async def get_action(self, obs):
        self.prompt_formatter.append_observation(obs)

        judge_messages = self.prompt_formatter.get_prompt("judge", self.current_goal, self.goal_age)
        judge_res = await self.generate(judge_messages)
        status, new_goal = self.parse_tag_content(judge_res["text"], ["status", "goal"])

        plan_completed = None
        adaptive_replan = None

        if status == "UPDATE" and new_goal:
            if self.current_goal != "[NO GOAL ASSIGNED]":
                if self.ask_for_plan_completion:
                    adherence_messages = build_plan_adherence_prompt(
                        self.current_goal, self.plan_observation_history, self.plan_action_history
                    )
                    comp_res = await self.generate(adherence_messages)
                    if "<eval>YES</eval>" in comp_res["text"]:
                        plan_completed = 1.0
                    elif "<eval>NO</eval>" in comp_res["text"]:
                        plan_completed = 0.0

                if self.ask_for_adaptive_replan:
                    adaptive_messages = build_adaptive_replan_system_prompt(
                        new_goal, self.plan_observation_history, self.plan_action_history
                    )
                    adapt_res = await self.generate(adaptive_messages)
                    if "<eval>YES</eval>" in adapt_res["text"]:
                        adaptive_replan = 1.0
                    elif "<eval>NO</eval>" in adapt_res["text"]:
                        adaptive_replan = 0.0

            self.current_goal = new_goal
            self.goal_age = 0

            self.plan_observation_history = []
            self.plan_action_history = []

            # Add the current observation back, matchine the plan_action_history
            self.plan_observation_history.append(obs)
            self.previous_plan = self.current_goal
        else:
            self.goal_age += 1

        actor_messages = self.prompt_formatter.get_prompt("actor", self.current_goal, self.goal_age)
        actor_res = await self.generate(actor_messages)
        action = self.parse_tag_content(actor_res["text"], ["action"])[0] or "noop"

        self.prompt_formatter.append_action(action)
        self.plan_action_history.append(action)

        return dict(
            action=action,
            plan=self.current_goal,
            plan_completed=plan_completed,
            adaptive_replan=adaptive_replan,
            logprob=actor_res["logprobs"],
            prompt_token_ids=actor_res["prompt_token_ids"],
            action_token_ids=actor_res["token_ids"],
            raw_output=f"<judge>\n{judge_res['text']}\n</judge>\n<actor>\n{actor_res['text']}\n</actor>",
        )


class HierarchicalSeparateNLEAgent(BaseAgent):
    def __init__(self, llm_judge, llm_actor, prompt_formatter, args: Args):
        super().__init__(llm_actor, prompt_formatter, args)
        self.llm_judge = llm_judge
        self.ask_for_plan_completion = args.ask_for_plan_completion
        self.ask_for_adaptive_replan = args.ask_for_adaptive_replan

        self.sampling_params_judge = vars(args.llm_judge.sampling_args)

    def reset(self):
        super().reset()
        self.current_goal = "[NO GOAL ASSIGNED]"
        self.goal_age = 0

        self.plan_action_history = []
        self.plan_observation_history = []
        self.previous_plan = None

    async def get_action(self, obs):
        self.prompt_formatter.append_observation(obs)
        self.plan_observation_history.append(obs)

        judge_messages = self.prompt_formatter.get_prompt("judge", self.current_goal, self.goal_age)
        # WE USE A SEPARATE LLM FOR JUDGE
        judge_res = await self.generate(judge_messages, self.llm_judge, self.sampling_params_judge)
        status, new_goal = self.parse_tag_content(judge_res["text"], ["status", "goal"])

        plan_completed = None
        adaptive_replan = None

        if status == "UPDATE" and new_goal:
            if self.current_goal != "[NO GOAL ASSIGNED]":
                if self.ask_for_plan_completion:
                    adherence_messages = build_plan_adherence_prompt(
                        self.current_goal, self.plan_observation_history, self.plan_action_history
                    )
                    comp_res = await self.generate(adherence_messages, self.llm_judge, self.sampling_params_judge)
                    if "<eval>YES</eval>" in comp_res["text"]:
                        plan_completed = 1.0
                    elif "<eval>NO</eval>" in comp_res["text"]:
                        plan_completed = 0.0

                if self.ask_for_adaptive_replan:
                    adaptive_messages = build_adaptive_replan_system_prompt(
                        new_goal, self.plan_observation_history, self.plan_action_history
                    )
                    adapt_res = await self.generate(adaptive_messages, self.llm_judge, self.sampling_params_judge)
                    if "<eval>YES</eval>" in adapt_res["text"]:
                        adaptive_replan = 1.0
                    elif "<eval>NO</eval>" in adapt_res["text"]:
                        adaptive_replan = 0.0

            self.current_goal = new_goal
            self.goal_age = 0

            self.plan_observation_history = []
            self.plan_action_history = []

            # Add the current observation back, matchine the plan_action_history
            self.plan_observation_history.append(obs)
            self.previous_plan = self.current_goal
        else:
            self.goal_age += 1

        actor_messages = self.prompt_formatter.get_prompt("actor", self.current_goal, self.goal_age)
        # WE USE THE DEFAULT LLM FOR ACTOR
        actor_res = await self.generate(actor_messages)
        action = self.parse_tag_content(actor_res["text"], ["action"])[0] or "noop"

        self.prompt_formatter.append_action(action)
        self.plan_action_history.append(action)

        return dict(
            action=action,
            plan=self.current_goal,
            plan_completed=plan_completed,
            adaptive_replan=adaptive_replan,
            logprob=actor_res["logprobs"],
            prompt_token_ids=actor_res["prompt_token_ids"],
            action_token_ids=actor_res["token_ids"],
            raw_output=f"<judge>\n{judge_res['text']}\n</judge>\n<actor>\n{actor_res['text']}\n</actor>",
        )
