def make_arc_agi_env(*args, **kwargs):
    from .make_env import make_arc_agi_env as _make_arc_agi_env

    return _make_arc_agi_env(*args, **kwargs)


__all__ = ["make_arc_agi_env"]
