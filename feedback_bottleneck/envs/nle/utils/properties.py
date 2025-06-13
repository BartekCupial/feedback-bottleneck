from __future__ import annotations

import enum

from nle import nethack as nh


class ItemCategory(enum.IntEnum):
    # Map NLE integer constants to an Enum for readability
    RANDOM = nh.RANDOM_CLASS
    ILLOBJ = nh.ILLOBJ_CLASS
    COIN = nh.COIN_CLASS
    AMULET = nh.AMULET_CLASS
    WEAPON = nh.WEAPON_CLASS
    ARMOR = nh.ARMOR_CLASS
    COMESTIBLES = nh.FOOD_CLASS
    SCROLL = nh.SCROLL_CLASS
    SPELLBOOK = nh.SPBOOK_CLASS
    POTION = nh.POTION_CLASS
    RING = nh.RING_CLASS
    WAND = nh.WAND_CLASS
    TOOL = nh.TOOL_CLASS
    GEM = nh.GEM_CLASS
    ROCK = nh.ROCK_CLASS
    BALL = nh.BALL_CLASS
    CHAIN = nh.CHAIN_CLASS
    VENOM = nh.VENOM_CLASS
    # Custom offsets for things handled differently in text
    CORPSE = nh.MAXOCLASSES + 1
    STATUE = nh.MAXOCLASSES + 2

    def __str__(self):
        return self.name.lower()


class ArmorClass(enum.IntEnum):
    SUIT = 0
    SHIELD = 1
    HELM = 2
    GLOVES = 3
    BOOTS = 4
    CLOAK = 5
    SHIRT = 6
