WIKI_TIPS = """
General Tips:
- Explore the environment to find the stairs down to the next level.
- NetHack rewards careful and methodical play. Take your time as rash decisions usually lead to an early death.
- Prioritize equipping reliable armor and weaponry, and upgrade as soon as better items become available.
- If you're weakened by hunger, pray for nutrition if your deity is receptive. Save permanent food (rations, tins, etc.) for emergencies when prayer isn't an option.
- Always read the last game message carefully to understand the current state and decide your next action.
- Read the language observation carefully and look at ascii map or image observation provided to decide the next action to take and where to move next.

Mines Specific Tips:
- Stairs to the mines lie on one of the dungeon levels 2-4.
- Avoid fighting too many monsters at once, escape upstairs or back off if not prepared.
- Kill dwarves to get a dwarvish mithril-coat.
- Don't dip items in fountains in mine town.
- Don't destroy walls in mine town.
- Don't kick doors in mine town.
- Don't attack peaceful monsters in mine town.
- Don't steal from shops.

Valkyrie Specific Tips:
- The starting long sword can be upgraded to Excalibur by dipping it in a fountain as soon as a lawful Valkyrie reaches experience level 7.
- There are always 4 fountains around the Oracle (levels 5-9), useful for Excalibur.
- By level 8 (typically XL 6-9) your gear should evolve from starting pieces to better loot found in the Mines. Your goal is AC -3 or better by the time you decide to venture lower into the dungeon.
"""

GAME_MECHANICS_TIPS = """
Game Mechanics Tips:
- You are the "@" symbol. Other "@" symbols may appear on the map. These are NPCs (humans, shopkeepers, guards, elves). Do not confuse them with yourself; read the message and cursor if necessary.
- When the message asks for a completion, such as: "What do you want to eat? [d or ?*]", you should respond with a single character corresponding to the item you want to eat/use.
    - For example, "What do you want to eat? [dgh or ?*]" -> Possible answers are "d", "g", or "h" to eat the associated food.
- When selecting multiple items (e.g., in a multi-pickup menu), after outputting the letters of the items you wish to select, you must output the action "more" to confirm the selection and close the menu.
- When the message asks for a direction, such as: "In what direction?" you should respond with a direction.
- When the message has --More-- at the end, your next action should be "more" to see the rest of the message.
- If you keep moving in the same direction, you will eventually hit a wall and stop moving. Your message might be: "It's solid stone", or "It's a wall". Change your action to move in another direction to continue exploring the environment.
- You can attack monsters by moving into them.
- If you are in a room without doors or corridor that seems like a dead end, use the 'search' command multiple times. Hidden doors and passages are common.
    - It is common to use a numeric prefix before the search command (e.g., "2" "2" "s") to maximize the chance of revealing adjacent hidden doors or passages.
    - One method of efficient searching to find a hidden door leading out of a room is to select a three-square range along the wall, type "2" "2" "s", then move three spaces along the wall and repeat until you find the door.
- Do not use the 'up' or 'upstairs' command on the first level (DLVL 1). Going up on the first level results in quitting the game and losing immediately. Only use 'down' to progress deeper.
"""
