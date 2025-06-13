def make_countdown_env(*args, **kwargs):
    from .make_env import make_countdown_env as _make_countdown_env

    return _make_countdown_env(*args, **kwargs)


__all__ = ["make_countdown_env"]
