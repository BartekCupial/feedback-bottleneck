import logging
import os
import pickle
import traceback
from pathlib import Path

import gymnasium as gym
from nle import nethack
from nle.env.base import NLE

from feedback_bottleneck.envs.utils import get_unique_seed

logger = logging.getLogger(__name__)


class SaveOnException(gym.Wrapper):
    def __init__(self, env, failed_game_path: str = None):
        super().__init__(env)

        self.failed_game_path = failed_game_path
        self.episode_number = 0

    def _get_obs_dict(self):
        """Helper to convert raw tuple observation to dict using env keys."""
        raw_obs = self.env.unwrapped.last_observation
        keys = self.env.unwrapped._observation_keys
        nle_obs = dict(zip(keys, raw_obs))

        return {
            "text": {},
            "image": None,
            "obs": nle_obs,
        }

    def reset(self, *, seed=None, **kwargs):
        self.recorded_seed = seed if seed is not None else get_unique_seed(episode_idx=self.episode_number)
        self.recorded_actions = []
        self.named_actions = []
        self.episode_number += 1

        try:
            return self.env.reset(seed=self.recorded_seed, **kwargs)
        except Exception as e:
            message = f"Bot failed due to unhandled exception: {str(e)}\n{traceback.format_exc()}"
            logger.info(message)
            self.save_to_file(message=message)

            obs = self._get_obs_dict()
            info = {}
            info["is_ascended"] = False
            info["end_status"] = NLE.StepStatus.ABORTED

            return obs, info

    def step(self, action):
        try:
            self.recorded_actions.append(action)
            return self.env.step(action)
        except Exception as e:
            message = f"Bot failed due to unhandled exception: {str(e)}\n{traceback.format_exc()}"
            logger.info(message)
            self.save_to_file(message=message)

            obs = self._get_obs_dict()
            reward = 0
            info = {}
            info["is_ascended"] = False
            info["end_status"] = NLE.StepStatus.ABORTED

            return obs, reward, True, False, info

    def save_to_file(self, message=""):
        dat = {
            "seed": self.recorded_seed,
            "actions": self.recorded_actions,
            "last_observation": self.env.unwrapped.last_observation,
            "message": message,
        }
        og_ttyrec = self.env.unwrapped.nethack._ttyrec
        if og_ttyrec is not None:
            ttyrec = Path(og_ttyrec).stem
        else:
            ttyrec_prefix = f"nle.{os.getpid()}.{self.recorded_seed}"
            ttyrec_version = f".ttyrec{nethack.TTYREC_VERSION}.bz2"
            ttyrec = ttyrec_prefix + ttyrec_version

        if not os.path.exists(self.failed_game_path):
            os.makedirs(self.failed_game_path, exist_ok=True)
        fname = os.path.join(self.failed_game_path, f"{ttyrec}.demo")
        with open(fname, "wb") as f:
            logger.info(f"Saving demo to {fname}...")
            pickle.dump(dat, f)
