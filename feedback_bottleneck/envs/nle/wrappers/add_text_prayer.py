import re

import gymnasium as gym

from feedback_bottleneck.envs.nle.utils.blstats import BLStats


class AddTextPrayer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.blstats = BLStats(*obs["blstats"])
        self.message = obs["text_message"]
        self.last_prayer = None
        self.angry = False

        return self.populate_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.blstats = BLStats(*obs["blstats"])
        self.message = obs["text_message"]
        self.update()

        return self.populate_obs(obs), reward, terminated, truncated, info

    def update(self):
        if "You finish your prayer." in self.message:
            self.last_prayer = self.blstats.time

        angry_regex = (
            r"(?:Thou hast angered me\."
            r"|Thou shalt pay, infidel\."
            r"|How darest thou desecrate my altar!"
            r"|Vile creature, thou durst call upon me\?"
            r"|Walk no more, perversion of nature!"
            r"|I believe it not!"
            r"|So, mortal! You dare desecrate my High Temple!"
            r"|Suffer, infidel!"
            r"|Thou must relearn thy lessons!"
            r"|Thou durst scorn me\?"
            r"|Thou durst call upon me\?"
            r"|You feel that .+ is displeased."
            r"|.+ is not deterred)"
        )

        if re.search(angry_regex, self.message):
            self.angry = True

        no_angry_regex = r"(?:have a feeling of reconciliation" r"|have a hopeful feeling" r"|seems mollified)"

        if re.search(no_angry_regex, self.message):
            self.angry = False

    def populate_obs(self, obs):
        return {**obs, "text_prayer": str(self)}

    def __str__(self):
        desc = []
        desc.append(
            f"Prayed {self.blstats.time - self.last_prayer} turn{'s' if self.blstats.time - self.last_prayer > 1 else ''} ago."
            if self.last_prayer
            else "Never prayed."
        )
        desc.append(f"God is {'not ' if not self.angry else ''}angry at you.")
        return "\n".join(desc)

    def __repr__(self):
        return self.__str__()
