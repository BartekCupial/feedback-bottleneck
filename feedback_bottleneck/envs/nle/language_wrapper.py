from typing import List

import gymnasium as gym

from feedback_bottleneck.envs.nle.actions import NLE_ACTION_MAP, NLE_DESCRIPTIONS
from feedback_bottleneck.envs.spaces import Strings


class NLELanguageWrapper(gym.Wrapper):
    def __init__(self, env, actions: List[str], provide_map: bool = True):
        super().__init__(env)

        self.provide_map = provide_map

        self.action_str_desc_map = {}
        self.action_str_enum_map = {}
        for action in actions:
            assert action in NLE_ACTION_MAP, f"Unknown action: {action}"
            assert action in NLE_DESCRIPTIONS, f"Unknown action: {action}"
            self.action_str_enum_map[action] = NLE_ACTION_MAP[action]
            self.action_str_desc_map[action] = NLE_DESCRIPTIONS[action]

        self.language_action_space = self.create_action_space()
        self.done = False
        self.max_steps = self.env.unwrapped._max_episode_steps

        if "esc" in actions:
            self.default_action = "esc"
        elif "north" in actions:
            self.default_action = "north"
        else:
            raise ValueError("No suitable default action found in action set.")

    def pre_reset(self):
        pass

    def reset(self, **kwargs):
        self.pre_reset()
        self.obs, self.info = self.env.reset(**kwargs)

        return self.post_reset(self.obs), self.info

    def post_reset(self, nle_obs):
        return self.nle_process_obs(nle_obs)

    def pre_step(self, action):
        """Translate language action to nle action."""
        if action not in self.action_str_enum_map:
            raise ValueError(f"Action {repr(action)} is not recognized " "or not supported for this environment")
        nle_action_enum = self.action_str_enum_map[action]
        nle_action_idx = self.env.actions.index(nle_action_enum)

        return nle_action_idx

    def step(self, action):
        action = self.pre_step(action)

        self.obs, reward, term, trun, self.info = self.env.step(action)

        return self.post_step(self.obs), reward, term, trun, self.info

    def post_step(self, nle_obs):
        return self.nle_process_obs(nle_obs)

    def get_text_action(self, action):
        return action

    def nle_process_obs(self, nle_obs):
        img = None
        text = self.render_hybrid(nle_obs) if self.provide_map else self.render_text(nle_obs)

        return {
            "text": text,
            "image": img,
            "obs": nle_obs,
        }

    def get_stats(self):
        return self.info.get("episode_extra_stats", {})

    def create_action_space(self):
        all_actions = list(self.action_str_desc_map.keys())
        return Strings(all_actions)

    def ascii_render(self, chars):
        rows, cols = chars.shape
        result = []
        for i in range(rows):
            result_row = ""
            for j in range(cols):
                entry = "<" + chr(chars[i, j]) + ">"
                result_row += entry
            result.append(result_row)
        return "\n".join(result)

    def render_text(self, nle_obs):
        long_term_observations = [
            ("text_message", "message"),
            ("text_cursor", "cursor"),
        ]

        short_term_observations = [
            ("text_overview", "overview"),
            ("text_map", "map description"),
            ("text_blstats", "blstats"),
            ("text_glyphs", "language observation"),
            ("text_prayer", "prayer status"),
            ("text_inventory", "inventory"),
        ]

        long_term_context = "\n".join([f"{name}:\n{nle_obs[key]}\n" for key, name in long_term_observations])
        short_term_context = "\n".join([f"{name}:\n{nle_obs[key]}\n" for key, name in short_term_observations])

        return {
            "long_term_context": long_term_context,
            "short_term_context": short_term_context,
        }

    def render_hybrid(self, nle_obs):
        # Logic:
        # 1. Take lines[1:-2] to remove the original top line and bottom two lines.
        # 2. Prepend [""] to act as the "empty" first line (index 0).
        # This ensures the map content starts at index 1, aligning with the actual Y coordinates.
        ascii_map = self.ascii_render(nle_obs["tty_chars"])
        lines = ascii_map.split("\n")
        ascii_map = "\n".join([""] + lines[1:-2])
        nle_obs["map"] = ascii_map

        cursor_pos = nle_obs["tty_cursor"]
        nle_obs["text_cursor"] = f"{nle_obs['text_cursor']} (x={cursor_pos[1]}, y={cursor_pos[0]})"

        long_term_observations = [
            ("text_message", "message"),
            ("text_cursor", "cursor"),
        ]

        short_term_observations = [
            ("text_overview", "overview"),
            ("text_map", "map description"),
            ("map", "map"),
            ("text_blstats", "blstats"),
            ("text_glyphs", "language observation"),
            ("text_prayer", "prayer status"),
            ("text_inventory", "inventory"),
        ]

        long_term_context = "\n".join([f"{name}:\n{nle_obs[key]}\n" for key, name in long_term_observations])
        short_term_context = "\n".join([f"{name}:\n{nle_obs[key]}\n" for key, name in short_term_observations])

        return {
            "long_term_context": long_term_context,
            "short_term_context": short_term_context,
        }
