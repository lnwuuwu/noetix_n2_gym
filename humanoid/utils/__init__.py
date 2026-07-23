"""Utility exports, loaded lazily so the PPO core can run without Isaac Gym.

Isaac Gym scripts keep the historical ``from humanoid.utils import ...`` API,
but importing a pure PyTorch algorithm no longer eagerly imports ``gymapi`` or
terrain utilities.  This is required by the native MuJoCo training entrypoint.
"""

from importlib import import_module


_HELPER_EXPORTS = {
    "class_to_dict",
    "get_load_path",
    "get_args",
    "export_policy_as_jit",
    "export_policy_as_onnx",
    "set_seed",
    "update_class_from_dict",
}
_MATH_EXPORTS = {
    "quat_apply_yaw",
    "wrap_to_pi",
    "torch_rand_sqrt_float",
}

__all__ = sorted(
    _HELPER_EXPORTS
    | _MATH_EXPORTS
    | {"task_registry", "Logger", "Terrain"}
)


def __getattr__(name):
    if name in _HELPER_EXPORTS:
        return getattr(import_module(".helpers", __name__), name)
    if name in _MATH_EXPORTS:
        return getattr(import_module(".math", __name__), name)
    if name == "task_registry":
        return import_module(".task_registry", __name__).task_registry
    if name == "Logger":
        return import_module(".logger", __name__).Logger
    if name == "Terrain":
        return import_module(".terrain", __name__).Terrain
    raise AttributeError("module {!r} has no attribute {!r}".format(
        __name__, name
    ))
