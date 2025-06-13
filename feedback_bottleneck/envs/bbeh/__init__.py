def make_bbeh_env(*args, **kwargs):
    from .make_env import make_bbeh_env as _make_bbeh_env

    return _make_bbeh_env(*args, **kwargs)


__all__ = ["make_bbeh_env"]
