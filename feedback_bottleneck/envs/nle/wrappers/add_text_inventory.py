import gymnasium as gym

from feedback_bottleneck.envs.nle.utils.inventory import Inventory


class AddTextInventory(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.inventory = Inventory()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.update(obs)
        return self.populate_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.update(obs)
        return self.populate_obs(obs), reward, terminated, truncated, info

    def update(self, obs):
        self.inventory.update(
            obs["inv_strs"],
            obs["inv_letters"],
            obs["inv_oclasses"],
            obs["inv_glyphs"],
        )

    def populate_obs(self, obs):
        return {
            **obs,
            "text_inventory": str(self.inventory),
        }
