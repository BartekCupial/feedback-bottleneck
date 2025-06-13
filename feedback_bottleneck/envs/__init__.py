from typing import Optional


def make_env(env_name, task, config, render_mode: Optional[str] = None, llm_judge=None):
    if env_name == "crafter":
        from feedback_bottleneck.envs.crafter.make_env import make_crafter_env

        return make_crafter_env(env_name, task, config, render_mode)
    elif env_name == "nle":
        from feedback_bottleneck.envs.nle.make_env import make_nle_env

        return make_nle_env(env_name, task, config, render_mode)
    elif env_name == "math":
        from feedback_bottleneck.envs.omni_math.make_env import make_math_env

        return make_math_env(env_name, task, config, render_mode, llm_judge=llm_judge)
    elif env_name == "countdown":
        from feedback_bottleneck.envs.countdown.make_env import make_countdown_env

        return make_countdown_env(env_name, task, config, render_mode, llm_judge=llm_judge)
    elif env_name == "bbeh":
        from feedback_bottleneck.envs.bbeh.make_env import make_bbeh_env

        return make_bbeh_env(env_name, task, config, render_mode, llm_judge=llm_judge)
    elif env_name == "sciknoweval":
        from feedback_bottleneck.envs.sciknoweval.make_env import make_sciknoweval_env

        return make_sciknoweval_env(env_name, task, config, render_mode, llm_judge=llm_judge)
    elif env_name == "enterprise_ops":
        from plan_crl.envs.enterprise_ops.make_env import make_enterprise_ops_env

        return make_enterprise_ops_env(env_name, task, config, render_mode, llm_judge=llm_judge)
    elif env_name in ("arc_agi", "arc_agi2"):
        from feedback_bottleneck.envs.arc_agi.make_env import make_arc_agi_env

        return make_arc_agi_env(env_name, task, config, render_mode, llm_judge=llm_judge)
    elif env_name == "code":
        from feedback_bottleneck.envs.omni_code.make_env import make_code_env

        return make_code_env(env_name, task, config, render_mode)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
