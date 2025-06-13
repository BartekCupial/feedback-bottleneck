import gymnasium as gym
import numpy as np
from nle import nethack, nle_language_obsv

from feedback_bottleneck.envs.nle.utils.blstats import BLStats
from feedback_bottleneck.envs.nle.utils.component_detection import get_revelable_positions, label_dungeon_features
from feedback_bottleneck.envs.nle.wrappers import AddTextMap


def is_wrapped(env, wrapper_class):
    current_env = env
    while hasattr(current_env, "env"):
        if isinstance(current_env, wrapper_class):
            return True
        current_env = current_env.env
    return isinstance(current_env, wrapper_class)


class AddTextObservation(gym.Wrapper):
    def __init__(self, env, crop_height=9, crop_width=9):
        super().__init__(env)

        assert is_wrapped(self.env, AddTextMap), "AddTextObservation requires AddTextMap."
        self.crop_h = crop_height
        self.crop_w = crop_width

        self.name_cache = {}
        self.translator = nle_language_obsv.NLELanguageObsv()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.populate_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.populate_obs(obs), reward, terminated, truncated, info

    def populate_obs(self, obs):
        new_obs = obs.copy()

        glyphs = obs["glyphs"]
        blstats = BLStats(*obs["blstats"])

        # Access persistent map data
        key = (blstats.dungeon_number, blstats.level_number)
        d_map = self.dungeon_maps[key]
        visited = d_map["visited"]
        objects = d_map["objects"]
        seen = d_map["seen"]

        labeled_features, _, _ = label_dungeon_features(objects)
        revelable_coords = get_revelable_positions(glyphs, seen, visited, labeled_features)

        revelable_mask = np.zeros(glyphs.shape, dtype=bool)
        if len(revelable_coords) > 0:
            revelable_mask[revelable_coords[:, 0], revelable_coords[:, 1]] = True

        new_obs["text_glyphs"] = self.translator.text_glyphs_with_mask(
            obs["glyphs"], obs["blstats"], revelable_mask
        ).decode("latin-1")

        # new_obs["text_glyphs"] = self.translator.text_glyphs(obs["glyphs"], obs["blstats"]).decode("latin-1")
        new_obs["text_blstats"] = self.translator.text_blstats(obs["blstats"]).decode("latin-1")
        new_obs["text_cursor"] = self.translator.text_cursor(obs["glyphs"], obs["blstats"], obs["tty_cursor"]).decode(
            "latin-1"
        )

        # Check for background object at player position
        bg_glyph = -1
        if 0 <= blstats.y < d_map["objects"].shape[0] and 0 <= blstats.x < d_map["objects"].shape[1]:
            bg_glyph = d_map["objects"][blstats.y, blstats.x]

        if bg_glyph != -1:
            name = self.translator.lookup_glyph(bg_glyph)

            ignored_surfaces = {
                "room floor",
                "dark room floor",
                "corridor floor",
                "lit corridor floor",
                "dark area",
                "vertical wall",
                "horizontal wall",
            }

            if name and name not in ignored_surfaces:
                new_obs["text_glyphs"] += f"\nYou are standing on {name}."

        return new_obs
