from typing import Optional

import crafter

from feedback_bottleneck.envs.crafter.language_wrapper import CrafterLanguageWrapper
from feedback_bottleneck.envs.env_wrapper import EnvWrapper
from feedback_bottleneck.envs.wrappers import GymV21CompatibilityV0, VideoRecorder


def make_crafter_env(env_name, task, config, render_mode: Optional[str] = None):
    env = crafter.Env(
        area=config.crafter_args.area,
        size=list(config.crafter_args.size),
        view=config.crafter_args.view,
        length=config.crafter_args.length,
        seed=config.seed,
    )
    if render_mode == "human":
        # Avoid importing pygame on headless cluster
        from feedback_bottleneck.envs.crafter.pygame_wrapper import PyGameWrapper

        env = PyGameWrapper(env)

    if config.language_wrapper:
        env = CrafterLanguageWrapper(
            env,
            task,
            max_episode_steps=env.unwrapped._length,
            unique_items=config.crafter_args.unique_items,
            precise_location=config.crafter_args.precise_location,
            skip_items=list(config.crafter_args.skip_items),
            edge_only_items=list(config.crafter_args.edge_only_items),
        )

    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    if config.language_wrapper:
        env = EnvWrapper(env, env_name, task)

    if config.record_videos and config.output_dir:
        env = VideoRecorder(env, config.output_dir)

    return env
