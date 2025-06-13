from __future__ import annotations

from nle import nethack as nh

from feedback_bottleneck.envs.nle.utils.properties import ArmorClass, ItemCategory


class Item:
    def __init__(self, letter: int, text: str, glyph: int, oclass: int):
        self.letter = letter
        self.text = text
        self.glyph = glyph
        self.oclass = oclass

        self.category = ItemCategory(oclass)

    @property
    def is_equipped(self) -> bool:
        return any(
            s in self.text for s in ["(being worn)", "(wielded)", "(in quiver)", "(on left hand)", "(on right hand)"]
        )

    @property
    def armor_class(self):
        """Returns the specific type of armor (Suit, Helm, etc) using NLE data."""
        if self.category == ItemCategory.ARMOR:
            # Convert glyph to internal object, then get armor category
            obj = nh.objclass(nh.glyph_to_obj(self.glyph))
            return ArmorClass(obj.oc_armcat)
        return None

    def __str__(self):
        return f"{chr(self.letter)}) {self.text}"

    def __repr__(self):
        return self.__str__()
