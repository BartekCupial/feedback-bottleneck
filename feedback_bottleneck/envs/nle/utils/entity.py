from typing import Any, Tuple

from nle import nethack


class Entity:
    def __init__(self, position: Tuple[int, int], glyph: int) -> None:
        """
        Entity class to represent a monster or player in the game.
        """
        self.position = position
        self.glyph = glyph
        self.permonst = self.get_permonst(glyph)
        self.name = self.permonst.mname if self.permonst else None
        self.difficulty = self.permonst.difficulty if self.permonst else None
        self.ac = self.permonst.ac if self.permonst else None
        self.cnutrit = self.permonst.cnutrit if self.permonst else None

    def get_permonst(self, glyph: int) -> str:
        """
        Get the monster name from the glyph.
        """
        if nethack.glyph_is_monster(glyph):
            mon_id = nethack.glyph_to_mon(glyph)
            mon = nethack.permonst(mon_id)
            return mon
        return None

    def __eq__(self, other: Any) -> bool:
        """
        Check if two entities are equal.
        """
        if isinstance(other, Entity):
            return self.position == other.position and self.glyph == other.glyph
        return False
