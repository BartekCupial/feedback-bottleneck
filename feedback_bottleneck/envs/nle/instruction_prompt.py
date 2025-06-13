from typing import List

from feedback_bottleneck.envs.nle.actions import MINIHACK_ACTIONS, NLE_ACTIONS, NLE_DESCRIPTIONS
from feedback_bottleneck.envs.nle.tips import GAME_MECHANICS_TIPS, WIKI_TIPS


def format_action_list(actions: List[str], descriptions: List[str]):
    """
    Groups actions to save tokens. Detailed descriptions for commands,
    but summarized for menu/character keys.
    """
    command_entries = []
    has_menu_keys = False

    for action, desc in zip(actions, descriptions):
        # Multi-character actions are primary commands (e.g., 'north', 'eat')
        # Single digits '0'-'9' are often used in menus too.
        if len(action) > 1:
            command_entries.append(f"{action}: {desc}")
        else:
            has_menu_keys = True

    action_strings = "\n".join(command_entries)

    if has_menu_keys:
        action_strings += "\n\n**Menu & Prompt Keys:**\n"
        action_strings += "Any single character (a-z, A-Z) or digit (0-9) is a valid action when responding to game prompts or selecting items from menus."

    return action_strings


def get_nle_prompt(
    task: str,
    actions: List[str],
    descriptions: List[str],
    use_wiki_tips: bool = True,
    use_game_mechanics_tips: bool = True,
):
    action_strings = format_action_list(actions, descriptions)

    instruction_prompt = f"""
You are an agent playing {task}. Your need to get as far as possible in the game.

CORE COMMANDS:
{action_strings}

{WIKI_TIPS.strip() if use_wiki_tips else ""}

{GAME_MECHANICS_TIPS.strip() if use_game_mechanics_tips else ""}

You will be presented with a history of interactions from the game.
""".strip()

    return instruction_prompt


def get_minihack_prompt(task: str, actions: List[str], descriptions: List[str]):
    if "Corridor" in task:
        goal = "Your need to explore the level and reach the stairs down."
    elif "Quest" in task:
        goal = "Your need to make use of an object is laying around for crossing a lava rivver (this can be any object allowing levitation or freezing), while fighting monsters and navigating rooms or mazes. Towards the end of the quests, you need to utlise a wand of death to kill a deadly monster guarding the stairs down."
    elif "Boxoban" in task:
        goal = "You are playing Boxoban, a box-pushing game inspired by Sokoban. Your need to push the boulders onto the fountains on the map. You can push the boulders by walking into them, as long as there are no obstacles behind them."
    elif "WoD" in task:
        goal = "You need to make use of wand of death to kill a deadly monster guarding the stairs down."
    else:
        goal = "Your need to get as far as possible in the game."

    action_strings = format_action_list(actions, descriptions)

    instruction_prompt = f"""
You are an agent playing {task}. {goal}

CORE COMMANDS:
{action_strings}.

You will be presented with a history of interactions from the game.
""".strip()

    return instruction_prompt


def get_instruction_prompt(task: str):
    if "minihack" in task.lower():
        actions = MINIHACK_ACTIONS
        descriptions = [NLE_DESCRIPTIONS[action] for action in actions]

        return get_minihack_prompt(task, actions, descriptions)
    elif "nethack" in task.lower():
        actions = NLE_ACTIONS
        descriptions = [NLE_DESCRIPTIONS[action] for action in actions]

        return get_nle_prompt(task, actions, descriptions)
    else:
        raise ValueError(f"Unknown NLE task: {task}")


def get_task_goal(task: str) -> str:
    if "MiniHack" in task:
        if "Corridor" in task:
            goal = "Your need to explore the level and reach the stairs down."
        elif "Quest" in task:
            goal = "Your need to make use of an object is laying around for crossing a lava rivver (this can be any object allowing levitation or freezing), while fighting monsters and navigating rooms or mazes. Towards the end of the quests, you need to utlise a wand of death to kill a deadly monster guarding the stairs down."
        elif "Boxoban" in task:
            goal = "You are playing Boxoban, a box-pushing game inspired by Sokoban. Your need to push the boulders onto the fountains on the map. You can push the boulders by walking into them, as long as there are no obstacles behind them."
        elif "WoD" in task:
            goal = "You need to make use of wand of death to kill a deadly monster guarding the stairs down."
        else:
            goal = "Your need to get as far as possible in the game."

    elif "NetHack" in task:
        goal = "Your need to get as far as possible in the game."

    else:
        raise ValueError(f"Unknown NLE task: {task}")

    return goal
