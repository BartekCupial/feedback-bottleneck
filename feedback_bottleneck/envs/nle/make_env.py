from typing import Optional

import gymnasium as gym
import minihack  # NOQA: F401
import nle  # NOQA: F401
import nle_progress  # NOQA: F401
from gymnasium.envs.registration import load_env_creator, spec
from nle import nethack
from nle_progress import NLEProgressWrapper
from nle_progress.task import NLEProgress

from feedback_bottleneck.envs.env_wrapper import EnvWrapper
from feedback_bottleneck.envs.nle.actions import MINIHACK_ACTIONS, NLE_ACTIONS
from feedback_bottleneck.envs.nle.language_wrapper import NLELanguageWrapper
from feedback_bottleneck.envs.nle.wrappers import (
    AddTextInventory,
    AddTextMap,
    AddTextObservation,
    AddTextOverview,
    AddTextPrayer,
    AutoMore,
    AutoRender,
    AutoSeed,
    FinalStats,
    NoProgressAbort,
    SaveOnException,
    TaskRewardsInfo,
)
from feedback_bottleneck.envs.wrappers import VideoRecorder


# We have to bypass gymnasium.make because it uses TimeLimit which conflicts with NLE
def nle_env_by_name(id: str):
    env_spec = spec(id)
    env_creator = load_env_creator(env_spec.entry_point)
    return env_creator


def make_nle_env(env_name, task, config, render_mode: Optional[str] = None):
    observation_keys = {
        "message",
        "blstats",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        "glyphs",
        "inv_strs",
        "inv_letters",
        "inv_glyphs",
        "inv_oclasses",
    }
    kwargs = dict(
        character=config.nle_args.character,
        max_episode_steps=config.nle_args.max_episode_steps,
        savedir=config.nle_args.savedir,
        save_ttyrec_every=config.nle_args.save_ttyrec_every,
        allow_all_yn_questions=config.nle_args.allow_all_yn_questions,
        allow_all_modes=config.nle_args.allow_all_modes,
        penalty_mode=config.nle_args.penalty_mode,
        penalty_step=config.nle_args.penalty_step,
        penalty_time=config.nle_args.penalty_time,
        observation_keys=observation_keys,
        actions=nethack.ACTIONS,
        render_mode=render_mode,
    )
    env_class = nle_env_by_name(task)
    env = env_class(**kwargs)
    env = AutoMore(env)
    env = AddTextMap(env)
    env = AddTextObservation(env)
    env = AddTextOverview(env)
    env = AddTextInventory(env)
    env = AddTextPrayer(env)
    env = NoProgressAbort(env, config.nle_args.no_progress_abort)
    env = AutoRender(env)
    env = AutoSeed(env)
    env = NLEProgressWrapper(env, progression_on_done_only=False)
    env = TaskRewardsInfo(env)
    env = FinalStats(env)

    if config.language_wrapper:
        if "minihack" in task.lower():
            actions = MINIHACK_ACTIONS
        elif "nethack" in task.lower():
            actions = NLE_ACTIONS
        else:
            raise ValueError(f"Unknown NLE task: {task}")

        env = NLELanguageWrapper(env, actions=actions, provide_map=config.nle_args.provide_map)
        env = EnvWrapper(env, env_name, task)

    if config.record_videos and config.output_dir:
        env = VideoRecorder(env, config.output_dir)

    env = SaveOnException(env, failed_game_path=config.output_dir)

    return env
