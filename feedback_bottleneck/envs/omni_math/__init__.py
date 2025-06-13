def make_math_env(*args, **kwargs):
    from .make_env import make_math_env as _make_math_env

    return _make_math_env(*args, **kwargs)


__all__ = ["make_math_env"]
