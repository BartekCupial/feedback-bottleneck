from __future__ import annotations

from typing import Dict, List, Optional

from feedback_bottleneck.envs.nle.utils.item import Item
from feedback_bottleneck.envs.nle.utils.properties import ArmorClass, ItemCategory


class Inventory:
    def __init__(self):
        self.items: Dict[int, Item] = {}

        # Define display grouping order
        self.groups = {
            "Coins": [ItemCategory.COIN],
            "Amulets": [ItemCategory.AMULET],
            "Weapons": [ItemCategory.WEAPON],
            "Armor": [ItemCategory.ARMOR],
            "Food": [ItemCategory.COMESTIBLES, ItemCategory.CORPSE],
            "Scrolls": [ItemCategory.SCROLL],
            "Spellbooks": [ItemCategory.SPELLBOOK],
            "Potions": [ItemCategory.POTION],
            "Rings": [ItemCategory.RING],
            "Wands": [ItemCategory.WAND],
            "Tools": [
                ItemCategory.TOOL,
                ItemCategory.GEM,
                ItemCategory.ROCK,
                ItemCategory.BALL,
                ItemCategory.CHAIN,
                ItemCategory.VENOM,
            ],
        }

    def update(self, inv_strs, inv_letters, inv_oclasses, inv_glyphs):
        """
        Rebuilds inventory from current NLE observation arrays.
        """
        self.items.clear()

        # NLE arrays are fixed size, 0 indicates end/empty
        for i, letter in enumerate(inv_letters):
            if letter == 0:
                break

            text = bytes(inv_strs[i]).decode("latin-1").strip("\0")

            self.items[letter] = Item(letter=letter, text=text, glyph=inv_glyphs[i], oclass=inv_oclasses[i])

    def __getitem__(self, key: str) -> List[Item]:
        """Get items by group name (e.g. 'Weapons')"""
        categories = self.groups.get(key, [])
        return [item for item in self.items.values() if item.category in categories]

    def __str__(self):
        """Generates the formatted text inventory."""
        output = []
        for group_name, categories in self.groups.items():
            # Find items belonging to this group
            group_items = [item for item in self.items.values() if item.category in categories]

            if group_items:
                output.append(f"{group_name}:")
                for item in group_items:
                    output.append(f"    {item}")

        return "\n".join(output) if output else "Inventory empty"

    @property
    def main_hand(self) -> Optional[Item]:
        # 'wielded' check in text
        for item in self["Weapons"] + self["Tools"]:
            if "(wielded)" in item.text:
                return item
        return None

    @property
    def worn_suit(self) -> Optional[Item]:
        return self._get_worn_armor(ArmorClass.SUIT)

    @property
    def worn_shield(self) -> Optional[Item]:
        return self._get_worn_armor(ArmorClass.SHIELD)

    @property
    def worn_helm(self) -> Optional[Item]:
        return self._get_worn_armor(ArmorClass.HELM)

    def _get_worn_armor(self, a_class: ArmorClass) -> Optional[Item]:
        for item in self["Armor"]:
            if item.is_equipped and item.armor_class == a_class:
                return item
        return None
