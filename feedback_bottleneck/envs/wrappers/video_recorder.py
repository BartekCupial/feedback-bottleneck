import datetime
from pathlib import Path

import gymnasium as gym
import imageio


class VideoRecorder(gym.Wrapper):
    def __init__(self, env, directory):
        super().__init__(env)

        if not hasattr(env, "episode_name"):
            env = EpisodeName(env)

        self.env = env
        self._directory = Path(directory).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._frames = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if obs["image"]:
            self._frames = [obs["image"]]

        return obs, info

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)
        if obs["image"]:
            self._frames.append(obs["image"])

        if term or trun:
            self._save()

        return obs, reward, term, trun, info

    def _save(self):
        if self._frames:
            filename = str(self._directory / (self.env.get_wrapper_attr("episode_name") + ".mp4"))
            imageio.mimsave(filename, self._frames)


class EpisodeName(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._timestamp = None
        self._unlocked = None
        self._seed = None
        self._length = 0
        self._reward = 0

    def reset(self, seed=None, **kwargs):
        self._seed = seed
        obs, info = self.env.reset(seed=seed, **kwargs)
        self._timestamp = None
        self._length = 0
        self._reward = 0

        return obs, info

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)
        self._length += 1
        self._reward += reward

        if term or trun:
            self._timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

        return obs, reward, term, trun, info

    @property
    def episode_name(self):
        return f"{self._timestamp}-seed{self._seed}-rew{self._reward:.2f}-len{self._length}"
