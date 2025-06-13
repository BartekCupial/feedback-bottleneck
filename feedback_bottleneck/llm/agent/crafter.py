import re
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from feedback_bottleneck.config.args import Args
from feedback_bottleneck.envs.crafter.actions import CRAFTER_ACTIONS, CRAFTER_DESCRIPTIONS
from feedback_bottleneck.llm.agent.base import BaseAgent, BaseFormatter


class XMLBuilder:
    @staticmethod
    def tag(name: str, content: Any) -> str:
        content_str = str(content).strip()
        return f"<{name}>\n{content_str}\n</{name}>"


# Most reliable list of Crafter achievements, actions, items etc.
# It shouldn't go out of sync with my Crafter repo since it's pretty stable
# https://github.com/danijar/crafter/blob/main/crafter/data.yaml
CRAFTER_ACHIEVEMENTS = [
    "Collect Wood",
    "Place Table",
    "Eat Cow",
    "Collect Sampling",
    "Collect Drink",
    "Make Wood Pickaxe",
    "Make Wood Sword",
    "Place Plant",
    "Defeat Zombie",
    "Collect Stone",
    "Place Stone",
    "Eat Plant",
    "Defeat Skeleton",
    "Make Stone Pickaxe",
    "Make Stone Sword",
    "Wake Up",
    "Place Furnace",
    "Collect Coal",
    "Collect Iron",
    "Make Iron Pickaxe",
    "Make Iron Sword",
    "Collect Diamond",
]


def _format_actions(actions: Sequence[str]) -> str:
    return "\n".join(f"- {action}: {CRAFTER_DESCRIPTIONS[action]}" for action in actions)


CRAFTER_ACTIONS_LIST = _format_actions(CRAFTER_ACTIONS)

CRAFTER_TIPS = """
- You can only act using the allowed actions list.
- Use the text state: it tells you what you face, what you see around, and your vitals/inventory.
- To collect materials / drink / attack, you generally need to face the target and use `Do`.
- If energy is low, `Sleep` can restore it (but you cannot act while sleeping).
- Prefer short, concrete subgoals you can finish soon; if a subgoal stalls, update it.
- Survival: if you feel unsafe or your vitals are low, build a simple enclosed stone shelter (leave a 1-tile entrance) near your table, then `Sleep` to restore energy safely.
- Exploration: systematically explore the whole map (sweep edges, then fill the interior). Exploration and mining are key for coal/iron/diamond and higher achievement counts.
- Iron tools: to craft iron tools, be next to a table and a furnace; place the furnace adjacent to the table.
- If you already have a table and furnace and at least some iron, try crafting an iron pickaxe quickly (it unlocks diamonds).
- Diamond: after you have an iron pickaxe, mine through stone to open new areas and keep exploring until you see diamond, then face it and `Do`.
- Food: if you need food and there is no cow nearby, eating a plant is acceptable (face plant and `Do`).

Progression ladder (recommended):
- Wood -> Place Table -> Make Wood Pickaxe (enables stone) -> Make Wood Sword (combat).
- Collect Stone -> Make Stone Pickaxe (enables coal/iron) -> Collect Coal -> Place Furnace.
- Collect Iron -> Make Iron Pickaxe -> Collect Diamond.
""".strip()


def _normalize_crafter_action(raw_action: str) -> str:
    if not raw_action:
        return raw_action

    raw = raw_action.strip()
    for valid in CRAFTER_ACTIONS:
        if raw.lower() == valid.lower():
            return valid

    cleaned = raw.lower().strip()
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
    cleaned = cleaned.replace("_", " ")
    cleaned = " ".join(cleaned.split())

    # LLMs like to spam those
    for prefix in ("final", "action:", "next:", "i will", "i choose", "action"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()

    # Map short, explicit movement commands (avoid matching random "west" in long analysis).
    if len(cleaned) <= 32:
        m = re.match(r"^(?:move|go|walk)?\s*(north|south|east|west|up|down|left|right)$", cleaned)
        if m:
            tok = m.group(1)
            tok = {"up": "north", "down": "south", "left": "west", "right": "east"}.get(tok, tok)
            candidate = f"Move {tok.title()}"
            if candidate in CRAFTER_ACTIONS:
                return candidate
        m = re.match(r"^(?:move|go|walk)\s+(north|south|east|west|up|down|left|right)$", cleaned)
        if m:
            tok = m.group(1)
            tok = {"up": "north", "down": "south", "left": "west", "right": "east"}.get(tok, tok)
            candidate = f"Move {tok.title()}"
            if candidate in CRAFTER_ACTIONS:
                return candidate

    if any(k in cleaned for k in ("noop", "wait", "idle", "nothing")):
        return "Noop"
    if any(k in cleaned for k in ("sleep", "rest")):
        return "Sleep"
    if cleaned in {"do", "interact", "use"}:
        return "Do"

    if "place" in cleaned or "build" in cleaned:
        for item, action in (
            ("stone", "Place Stone"),
            ("table", "Place Table"),
            ("furnace", "Place Furnace"),
            ("plant", "Place Plant"),
        ):
            if item in cleaned and action in CRAFTER_ACTIONS:
                return action

    if "make" in cleaned or "craft" in cleaned:
        material = None
        for m in ("wood", "stone", "iron"):
            if m in cleaned:
                material = m
                break
        tool = None
        for t in ("pickaxe", "sword"):
            if t in cleaned:
                tool = t
                break
        if material and tool:
            candidate = f"Make {material.title()} {tool.title()}"
            for valid in CRAFTER_ACTIONS:
                if candidate.lower() == valid.lower():
                    return valid

    # If we can't map it, return raw and let EnvWrapper handle validity/correction
    return raw


def _parse_vitals(inventory_text: str) -> Dict[str, int]:
    vitals: Dict[str, int] = {}
    for key in ("health", "food", "drink", "energy"):
        m = re.search(rf"-\s*{key}:\s*(\d+)\s*/\s*(\d+)", inventory_text, re.IGNORECASE)
        if m:
            vitals[key] = int(m.group(1))
    return vitals


def _parse_inventory_items(inventory_text: str) -> Dict[str, int]:
    items: Dict[str, int] = {}
    for m in re.finditer(r"^\s*-\s*([a-zA-Z_]+)\s*:\s*(\d+)\s*$", inventory_text, re.MULTILINE):
        name = m.group(1).strip().lower()
        count = int(m.group(2))
        items[name] = count
    return items


def _parse_front_object(world_text: str) -> str:
    m = re.search(r"^You face\s+(.*?)\s+at your front\.$", world_text, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return ""


def _parse_visible_objects(world_text: str) -> List[Tuple[str, int, int]]:
    """Parse lines like '- tree 3 steps north and 4 steps west' into (name, dy, dx)."""
    results: List[Tuple[str, int, int]] = []
    for line in world_text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        line = line[2:].strip()

        # '- tree 4 steps west'
        m1 = re.match(r"^(?P<obj>.+?)\s+(?P<n1>\d+)\s+steps?\s+(?P<d1>north|south|east|west)$", line, re.I)
        # '- tree 3 steps north and 4 steps west'
        m2 = re.match(
            r"^(?P<obj>.+?)\s+(?P<n1>\d+)\s+steps?\s+(?P<d1>north|south|east|west)\s+and\s+(?P<n2>\d+)\s+steps?\s+(?P<d2>north|south|east|west)$",
            line,
            re.I,
        )

        dy = 0
        dx = 0
        obj = ""
        parts: List[Tuple[int, str]] = []
        # m2 first since m1 could omit something, but we are not guaranteed to get m2 (or even m1)
        if m2:
            obj = m2.group("obj").strip().lower()
            parts = [(int(m2.group("n1")), m2.group("d1").lower()), (int(m2.group("n2")), m2.group("d2").lower())]
        elif m1:
            obj = m1.group("obj").strip().lower()
            parts = [(int(m1.group("n1")), m1.group("d1").lower())]
        else:
            continue

        for n, d in parts:
            if d == "north":
                dy += n
            elif d == "south":
                dy -= n
            elif d == "east":
                dx += n
            elif d == "west":
                dx -= n

        results.append((obj, dy, dx))

    return results


def _goal_targets(goal: str) -> List[str]:
    g = (goal or "").lower()
    # early-game achievements
    if "wood" in g:
        return ["tree"]
    if "drink" in g:
        return ["water", "river"]
    if "cow" in g:
        return ["cow"]
    if "zombie" in g:
        return ["zombie"]
    if "skeleton" in g:
        return ["skeleton"]
    if "plant" in g:
        return ["plant"]
    if "stone" in g:
        return ["stone"]
    if "coal" in g:
        return ["coal"]
    if "iron" in g:
        return ["iron"]
    if "diamond" in g:
        return ["diamond"]
    return []


def _detect_table(world: str) -> bool:
    w = (world or "").lower()
    return "table" in w


def _nearest_visible(world: str, needle: str) -> Tuple[int, int, int] | None:
    needle_l = (needle or "").lower()
    best = None
    for obj, dy, dx in _parse_visible_objects(world):
        if needle_l not in obj:
            continue
        dist = abs(dy) + abs(dx)
        if dist == 0:
            continue
        if best is None or dist < best[0]:
            best = (dist, dy, dx)
    return best


def _suggest_goal(*, world: str, inventory: str) -> str:
    items = _parse_inventory_items(inventory)
    wood = items.get("wood", 0)
    has_table = _detect_table(world)

    # Table is usually the first crafting unlock
    if not has_table and wood >= 2:
        return "Place Table"
    if not has_table:
        return "Collect Wood"

    # After table, craft basic weapon/tool once
    if items.get("wood_sword", 0) <= 0 and wood >= 2:
        return "Make Wood Sword"
    if items.get("wood_pickaxe", 0) <= 0 and wood >= 2:
        return "Make Wood Pickaxe"
    return "Collect Wood"


def _pick_exploration_move(act_history: Sequence[str]) -> str:
    # Prefer actions that were not used recently to avoid getting stuck
    moves = ["Move North", "Move East", "Move South", "Move West"]
    recent = list(act_history)[-8:]
    scores = []
    for m in moves:
        try:
            last = len(recent) - 1 - recent[::-1].index(m)
        except ValueError:
            last = -1
        scores.append((last, m))
    scores.sort(key=lambda x: x[0])
    return scores[0][1]


def _heuristic_action(*, goal: str, world: str, inventory: str, act_history: Sequence[str]) -> str:
    vitals = _parse_vitals(inventory)
    items = _parse_inventory_items(inventory)
    if vitals.get("energy", 9) <= 2:
        return "Sleep"

    g = (goal or "").lower()
    front = _parse_front_object(world)
    has_table = _detect_table(world)

    # Place table once we can afford it
    if "place table" in g and not has_table and items.get("wood", 0) >= 2:
        # Place into an empty tile in front.
        if front in {"", "nothing"}:
            return "Place Table"
        return _pick_exploration_move(act_history)

    if "make wood sword" in g and has_table:
        # Need to be near a table for crafting; otherwise move towards it
        table = _nearest_visible(world, "table")
        if table and table[0] > 1:
            _, dy, dx = table
            if abs(dy) >= abs(dx) and dy != 0:
                return "Move North" if dy > 0 else "Move South"
            if dx != 0:
                return "Move East" if dx > 0 else "Move West"
        if (
            table
            and table[0] <= 1
            and items.get("wood", 0) >= 2
            and items.get("wood_sword", 0) <= 0
            and list(act_history)[-3:] != ["Make Wood Sword"] * 3
        ):
            return "Make Wood Sword"

    if "make wood pickaxe" in g and has_table:
        table = _nearest_visible(world, "table")
        if table and table[0] > 1:
            _, dy, dx = table
            if abs(dy) >= abs(dx) and dy != 0:
                return "Move North" if dy > 0 else "Move South"
            if dx != 0:
                return "Move East" if dx > 0 else "Move West"
        if (
            table
            and table[0] <= 1
            and items.get("wood", 0) >= 2
            and items.get("wood_pickaxe", 0) <= 0
            and list(act_history)[-3:] != ["Make Wood Pickaxe"] * 3
        ):
            return "Make Wood Pickaxe"

    targets = _goal_targets(goal)
    visibles = _parse_visible_objects(world)
    if targets:
        if any(t in front for t in targets):
            return "Do"

    blocked_moves = set()
    for obj, dy, dx in visibles:
        if abs(dy) + abs(dx) != 1:
            continue
        # Any object adjacent blocks movement into that tile - trees/mobs etc.
        if dy == 1 and dx == 0:
            blocked_moves.add("Move North")
        elif dy == -1 and dx == 0:
            blocked_moves.add("Move South")
        elif dy == 0 and dx == 1:
            blocked_moves.add("Move East")
        elif dy == 0 and dx == -1:
            blocked_moves.add("Move West")

    # Choose the nearest visible target object
    best = None
    for obj, dy, dx in visibles:
        if targets and not any(t in obj for t in targets):
            continue
        dist = abs(dy) + abs(dx)
        if dist <= 0:
            continue
        if best is None or dist < best[0]:
            best = (dist, dy, dx)

    if best is None:
        return _pick_exploration_move(act_history)

    _, dy, dx = best
    # Move along the largest remaining axis first
    if abs(dy) >= abs(dx) and dy != 0:
        primary = "Move North" if dy > 0 else "Move South"
        if primary in blocked_moves and dx != 0:
            secondary = "Move East" if dx > 0 else "Move West"
            if secondary not in blocked_moves:
                return secondary
        if primary not in blocked_moves:
            return primary
        return _pick_exploration_move(act_history)
    if dx != 0:
        primary = "Move East" if dx > 0 else "Move West"
        if primary in blocked_moves and dy != 0:
            secondary = "Move North" if dy > 0 else "Move South"
            if secondary not in blocked_moves:
                return secondary
        if primary not in blocked_moves:
            return primary
        return _pick_exploration_move(act_history)
    return _pick_exploration_move(act_history)


def build_observation_block(obs_history: Sequence[Dict[str, Any]]) -> str:
    last = obs_history[-1]
    text = last.get("text", {}) or {}
    world = text.get("long_term_context", "")
    inventory = text.get("short_term_context", "")

    current = "\n".join(
        [
            XMLBuilder.tag("world", world),
            XMLBuilder.tag("inventory", inventory),
        ]
    )

    # History is lightweight - last few snapshots
    history_entries: List[str] = []
    for h in list(obs_history)[-4:]:
        h_text = h.get("text", {}) or {}
        h_world = h_text.get("long_term_context", "")
        h_inv = h_text.get("short_term_context", "")
        history_entries.append(
            XMLBuilder.tag("step", XMLBuilder.tag("world", h_world) + "\n" + XMLBuilder.tag("inventory", h_inv))
        )

    recent = XMLBuilder.tag("recent_steps", "\n".join(history_entries))
    return XMLBuilder.tag("current_state", current + "\n" + recent)


def build_actions_history_block(act_history: Sequence[str]) -> str:
    return XMLBuilder.tag("action_history", "\n".join(act_history))


def build_judge_system_prompt(*, task_goal: str) -> str:
    persona = "You are an expert Crafter coach. You guide a beginner agent by maintaining a good short-term subgoal."

    instructions = [
        "Read <current_state>, <action_history>, and <current_subgoal>.",
        "Decide if the current subgoal should be kept (KEEP) or replaced (UPDATE).",
        "UPDATE when the subgoal is completed, impossible, unsafe, or not making progress.",
        "If UPDATE, write a new subgoal that is specific and achievable within ~10-30 steps.",
        "Prefer subgoals that improve survival (vitals) and unlock achievements efficiently.",
        "Before/during night, prioritize subgoals that improve survival (shelter + sleeping) over risky exploration.",
        "Choose the new subgoal from <achievements> OR from a Crafter action name (e.g. Place Furnace, Make Iron Pickaxe).",
        "Return ONLY XML with two tags: <status> and <goal> (goal empty if KEEP).",
        "Do not include any other text outside the XML tags.",
    ]

    rules = XMLBuilder.tag("task_instruction", "\n".join(f"- {i}" for i in instructions))
    tips = XMLBuilder.tag("tips", CRAFTER_TIPS)
    achievements = XMLBuilder.tag("achievements", "\n".join(f"- {a}" for a in CRAFTER_ACHIEVEMENTS))
    goal = XMLBuilder.tag("overall_goal", task_goal)

    fmt = XMLBuilder.tag(
        "output_format",
        "\n".join(
            [
                "<status>KEEP</status>",
                "<goal></goal>",
                "<status>UPDATE</status>",
                "<goal>Collect Wood</goal>",
            ]
        ),
    )

    return "\n\n".join([persona, goal, achievements, rules, tips, fmt])


def build_hierarchical_actor_system_prompt(*, task_goal: str) -> str:
    persona = "You are a Crafter agent. Follow the current subgoal and choose a single best next action."

    instructions = [
        "Read <current_state>, <action_history>, and <current_subgoal>.",
        "Choose the single best next action from <allowed_actions> to advance the subgoal.",
        "Your response MUST contain exactly one action line in the form: <action>: <one action from the allowed list> </action>.",
        "Put the action tag as the last line of your response (nothing after it).",
        "If uncertain, prefer exploration over waiting: output <action> Move North </action>",
    ]

    rules = XMLBuilder.tag("task_instruction", "\n".join(f"- {i}" for i in instructions))
    goal = XMLBuilder.tag("overall_goal", task_goal)
    allowed = XMLBuilder.tag("allowed_actions", CRAFTER_ACTIONS_LIST)
    fmt = XMLBuilder.tag(
        "output_format",
        "\n".join(
            [
                "<action> Move West </action>",
                "<action> Do </action>",
                "<action> Noop </action>",
            ]
        ),
    )

    return "\n\n".join([persona, goal, rules, XMLBuilder.tag("tips", CRAFTER_TIPS), allowed, fmt])


def build_naive_actor_system_prompt(task_goal: str) -> str:
    persona = "You are a Crafter agent. Follow the current subgoal and choose a single best next action."

    instructions = [
        "Read <current_state>, and <action_history>.",
        "Choose the best immediate action from <allowed_actions>.",
        "Provide ONLY the action command inside <action> tags.",
        "If uncertain, prefer exploration over waiting: output <action> Move North </action>",
    ]

    rules = XMLBuilder.tag("task_instruction", "\n".join(f"- {i}" for i in instructions))
    goal = XMLBuilder.tag("overall_goal", task_goal)
    allowed = XMLBuilder.tag("allowed_actions", CRAFTER_ACTIONS_LIST)
    fmt = XMLBuilder.tag(
        "output_format",
        "\n".join(
            [
                "<action> Move West </action>",
                "<action> Do </action>",
                "<action> Noop </action>",
            ]
        ),
    )

    return "\n\n".join([persona, goal, rules, XMLBuilder.tag("tips", CRAFTER_TIPS), allowed, fmt])


def build_subgoal_block(current_goal: str, goal_age: int) -> str:
    content = [current_goal, XMLBuilder.tag("goal_age", goal_age)]
    if goal_age > 30:
        content.append(XMLBuilder.tag("warning", "Subgoal is stagnating. Consider UPDATE."))
    return XMLBuilder.tag("current_subgoal", "\n".join(map(str, content)))


class NaiveCrafterFormatter(BaseFormatter):
    def get_prompt(self, role: str = "actor", **kwargs) -> List[Dict[str, str]]:
        task_goal = "Complete as many Crafter achievements as possible (aim for steady, safe progress)."

        sys_text = build_naive_actor_system_prompt(task_goal)
        user_text = "\n\n".join(
            [build_actions_history_block(self.act_history), build_observation_block(self.obs_history)]
        )

        return [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

    def generate_sft_samples(self, episode_iterator):
        self.reset()

        for step_data in episode_iterator:
            crafter_obs = {}
            for key, val in step_data.items():
                if key.startswith("obs_"):
                    clean_key = key[4:]
                    if isinstance(val, list):
                        crafter_obs[clean_key] = np.asanyarray(val)
                    else:
                        crafter_obs[clean_key] = val

            observation = {
                "obs": crafter_obs,
                "short_term_context": step_data["short_term_context"],
                "long_term_context": step_data["long_term_context"],
                # Crafter specific mapping for build_observation_block
                "text": {
                    "short_term_context": step_data["short_term_context"],
                    "long_term_context": step_data["long_term_context"],
                },
            }
            self.append_observation(observation)

            messages = self.get_prompt(role="actor")
            action_text = step_data["text_actions"]
            target = XMLBuilder.tag("action", action_text)
            yield messages, target

            self.append_action(action_text)


class HierarchicalCrafterFormatter(BaseFormatter):
    def get_prompt(self, role: str, current_goal: str, goal_age: int) -> List[Dict[str, str]]:
        task_goal = "Complete as many Crafter achievements as possible (aim for steady, safe progress)."

        if role == "judge":
            sys_text = build_judge_system_prompt(task_goal=task_goal)
        else:
            sys_text = build_hierarchical_actor_system_prompt(task_goal=task_goal)

        user_text = "\n\n".join(
            [
                build_subgoal_block(current_goal, goal_age),
                build_actions_history_block(self.act_history),
                build_observation_block(self.obs_history),
            ]
        )

        return [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

    def generate_sft_samples(self, episode_iterator):
        self.reset()
        current_goal = "Collect Wood"
        goal_age = 0

        for step_data in episode_iterator:
            new_goal = step_data.get("plans", current_goal)

            if new_goal != current_goal:
                current_goal = new_goal
                goal_age = 0
            else:
                goal_age += 1

            crafter_obs = {}
            for key, val in step_data.items():
                if key.startswith("obs_"):
                    clean_key = key[4:]
                    if isinstance(val, list):
                        crafter_obs[clean_key] = np.asanyarray(val)
                    else:
                        crafter_obs[clean_key] = val

            observation = {
                "obs": crafter_obs,
                "short_term_context": step_data["short_term_context"],
                "long_term_context": step_data["long_term_context"],
                # Crafter specific mapping for build_observation_block
                "text": {
                    "short_term_context": step_data["short_term_context"],
                    "long_term_context": step_data["long_term_context"],
                },
            }
            self.append_observation(observation)

            messages = self.get_prompt(
                role="actor",
                current_goal=current_goal,
                goal_age=goal_age,
            )
            action_text = step_data["text_actions"]
            target = XMLBuilder.tag("action", action_text)
            yield messages, target

            self.append_action(action_text)


class NaiveCrafterAgent(BaseAgent):
    async def get_action(self, obs):
        self.prompt_formatter.append_observation(obs)

        messages = self.prompt_formatter.get_prompt(role="actor")
        output = await self.generate(messages)
        action_raw = self.parse_tag_content(output["text"], ["action"])[0]

        action = _normalize_crafter_action(action_raw) or "noop"
        if action not in CRAFTER_ACTIONS:
            action = "noop"

        self.prompt_formatter.append_action(action)

        return dict(
            action=action,
            plan="",
            logprob=output["logprobs"],
            prompt_token_ids=output["prompt_token_ids"],
            action_token_ids=output["token_ids"],
            raw_output=output["text"],
        )


class HierarchicalCrafterAgent(BaseAgent):
    def reset(self):
        super().reset()
        self.current_goal = "Collect Wood"
        self.goal_age = 0

    async def get_action(self, obs):
        self.prompt_formatter.append_observation(obs)

        judge_messages = self.prompt_formatter.get_prompt("judge", self.current_goal, self.goal_age)
        judge_res = await self.generate(judge_messages)
        status, new_goal = self.parse_tag_content(judge_res["text"], ["status", "goal"])

        if status == "UPDATE" and new_goal:
            self.current_goal = new_goal
            self.goal_age = 0
        else:
            self.goal_age += 1

        actor_messages = self.prompt_formatter.get_prompt("actor", self.current_goal, self.goal_age)
        actor_res = await self.generate(actor_messages)
        action = self.parse_tag_content(actor_res["text"], ["action"])[0] or "noop"

        self.prompt_formatter.append_action(action)

        return dict(
            action=action,
            plan=self.current_goal,
            logprob=actor_res["logprobs"],
            prompt_token_ids=actor_res["prompt_token_ids"],
            action_token_ids=actor_res["token_ids"],
            raw_output=f"<judge>\n{judge_res['text']}\n</judge>\n<actor>\n{actor_res['text']}\n</actor>",
        )
