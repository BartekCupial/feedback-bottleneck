from feedback_bottleneck.envs.crafter.actions import CRAFTER_DESCRIPTIONS


def get_instruction_prompt(task=None):
    action_strings = ",\n".join(f"{action}: {description}" for action, description in CRAFTER_DESCRIPTIONS.items())
    instruction_prompt = f"""
You are an agent playing Crafter. The following are the only valid actions you can take in the game, followed by a short description of each action:

{action_strings}.

These are the game achievements you can get:
1. Collect Wood
2. Place Table
3. Eat Cow
4. Collect Sampling
5. Collect Drink
6. Make Wood Pickaxe
7. Make Wood Sword
8. Place Plant
9. Defeat Zombie
10. Collect Stone
11. Place Stone
12. Eat Plant
13. Defeat Skeleton
14. Make Stone Pickaxe
15. Make Stone Sword
16. Wake Up
17. Place Furnace
18. Collect Coal
19. Collect Iron
20. Make Iron Pickaxe
21. Make Iron Sword
22. Collect Diamond

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible by completing all the achievements.

PLAY!
""".strip()

    return instruction_prompt
