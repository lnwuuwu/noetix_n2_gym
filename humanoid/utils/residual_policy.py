"""Checkpoint/config helpers for the bounded stair residual policy."""

from __future__ import annotations

import os

import torch

from humanoid import LEGGED_GYM_ROOT_DIR


DEFAULT_BASE_OBS_DIM = 410
DEFAULT_TARGET_OBS_DIM = 15
DEFAULT_HIDDEN_DIMS = [128, 64]
DEFAULT_ACTION_INDICES = [4, 5, 6, 7, 8, 13, 14, 15, 16, 17]
DEFAULT_ACTION_SCALES = [
    0.08,
    0.12,
    0.16,
    0.20,
    0.12,
    0.08,
    0.12,
    0.16,
    0.20,
    0.12,
]
DEFAULT_FROZEN_ACTION_STD = 1.0e-4


def resolve_checkpoint_path(args, train_cfg):
    """Resolve the exact checkpoint selected by standard runner arguments."""
    # Import lazily: helpers also owns Isaac Gym argument parsing, while
    # checkpoint metadata inspection is intentionally CPU/test friendly.
    from humanoid.utils.helpers import get_load_path

    direct = getattr(args, "model_path", None)
    if direct:
        path = os.path.abspath(os.path.expanduser(direct))
        if not os.path.isfile(path):
            raise FileNotFoundError("Checkpoint does not exist: " + path)
        return path
    root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
    )
    load_run = (
        args.load_run
        if getattr(args, "load_run", None) is not None
        else train_cfg.runner.load_run
    )
    checkpoint = (
        args.checkpoint
        if getattr(args, "checkpoint", None) is not None
        else train_cfg.runner.checkpoint
    )
    return get_load_path(root, load_run=load_run, checkpoint=checkpoint)


def _load_checkpoint(path):
    try:
        return torch.load(path, weights_only=False, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def residual_metadata_from_checkpoint(path):
    """Return residual policy metadata, inferring old metadata if necessary."""
    checkpoint = _load_checkpoint(path)
    metadata = checkpoint.get("policy_metadata")
    if (
        isinstance(metadata, dict)
        and metadata.get("class_name") == "ResidualActorCritic"
    ):
        return dict(metadata)

    state = checkpoint.get("model_state_dict", {})
    residual_weights = []
    for key, value in state.items():
        if (
            key.startswith("residual_actor.")
            and key.endswith(".weight")
            and isinstance(value, torch.Tensor)
            and value.ndim == 2
        ):
            layer_index = int(key.split(".")[1])
            residual_weights.append((layer_index, value))
    if not residual_weights:
        return None
    residual_weights.sort(key=lambda item: item[0])
    base_weight = state.get("actor.0.weight")
    if base_weight is None:
        raise RuntimeError(
            "Residual checkpoint is missing base Actor input weights: " + path
        )
    action_indices = state.get("residual_action_indices")
    action_scales = state.get("residual_action_scales")
    frozen_std = state.get("residual_frozen_action_std")
    if action_indices is None or action_scales is None:
        raise RuntimeError(
            "Residual checkpoint is missing action safety buffers: " + path
        )
    return {
        "class_name": "ResidualActorCritic",
        "residual_base_obs_dim": int(base_weight.shape[1]),
        "residual_observation_dim": int(residual_weights[0][1].shape[1]),
        "residual_hidden_dims": [
            int(weight.shape[0]) for _, weight in residual_weights[:-1]
        ],
        "residual_action_indices": action_indices.tolist(),
        "residual_action_scales": action_scales.tolist(),
        "residual_frozen_action_std": (
            float(frozen_std.item())
            if frozen_std is not None
            else DEFAULT_FROZEN_ACTION_STD
        ),
    }


def residual_metadata(
    hidden_dims=None,
    action_scales=None,
    base_obs_dim=DEFAULT_BASE_OBS_DIM,
    target_obs_dim=DEFAULT_TARGET_OBS_DIM,
):
    """Build validated metadata for conversion from a normal checkpoint."""
    hidden_dims = list(hidden_dims or DEFAULT_HIDDEN_DIMS)
    action_scales = list(action_scales or DEFAULT_ACTION_SCALES)
    if len(action_scales) != len(DEFAULT_ACTION_INDICES):
        raise ValueError(
            "Residual action scales must contain {} values".format(
                len(DEFAULT_ACTION_INDICES)
            )
        )
    return {
        "class_name": "ResidualActorCritic",
        "residual_base_obs_dim": int(base_obs_dim),
        "residual_observation_dim": int(base_obs_dim) + int(target_obs_dim),
        "residual_hidden_dims": [int(value) for value in hidden_dims],
        "residual_action_indices": list(DEFAULT_ACTION_INDICES),
        "residual_action_scales": [float(value) for value in action_scales],
        "residual_frozen_action_std": DEFAULT_FROZEN_ACTION_STD,
    }


def configure_residual_policy(env_cfg, train_cfg, metadata):
    """Mutate the selected process-local configs to match a checkpoint."""
    if metadata is None:
        return False
    base_obs_dim = int(metadata["residual_base_obs_dim"])
    observation_dim = int(metadata["residual_observation_dim"])
    target_obs_dim = observation_dim - base_obs_dim
    configured_base = (
        int(env_cfg.env.frame_stack) * int(env_cfg.env.num_single_obs)
    )
    if configured_base != base_obs_dim:
        raise RuntimeError(
            "Residual checkpoint base observation width {} does not match "
            "task width {}".format(base_obs_dim, configured_base)
        )
    if target_obs_dim != int(
        getattr(env_cfg.env, "residual_target_obs_dim", target_obs_dim)
    ):
        raise RuntimeError(
            "Residual checkpoint target width {} does not match task width {}"
            .format(
                target_obs_dim,
                getattr(env_cfg.env, "residual_target_obs_dim", None),
            )
        )
    env_cfg.env.include_residual_targets = True
    env_cfg.env.residual_base_obs_dim = base_obs_dim
    env_cfg.env.residual_target_obs_dim = target_obs_dim
    env_cfg.env.num_observations = observation_dim

    train_cfg.policy.class_name = "ResidualActorCritic"
    train_cfg.policy.residual_base_obs_dim = base_obs_dim
    train_cfg.policy.residual_hidden_dims = list(
        metadata["residual_hidden_dims"]
    )
    train_cfg.policy.residual_action_indices = list(
        metadata["residual_action_indices"]
    )
    train_cfg.policy.residual_action_scales = list(
        metadata["residual_action_scales"]
    )
    train_cfg.policy.residual_frozen_action_std = float(
        metadata.get(
            "residual_frozen_action_std", DEFAULT_FROZEN_ACTION_STD
        )
    )
    return True
