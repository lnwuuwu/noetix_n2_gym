"""Shared N2 AMP observation construction.

The discriminator and reference-motion collector must use exactly the same
coordinate frame and feature order.  Keeping two handwritten implementations
previously produced world-frame feet for the policy and misleading zero padding
for half of the reference key-point vector.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


AMP_FEATURE_VERSION = "n2_amp_v2_end_effectors_local"
AMP_OBSERVATION_DIM = 55
AMP_FULL_FRAME_DIM = 62
AMP_MASKED_OBSERVATION_INDICES = (8, 17, 44, 53)

ROOT_POSITION = slice(0, 3)
ROOT_ROTATION = slice(3, 7)
JOINT_POSITION = slice(7, 25)
KEY_POSITION = slice(25, 37)
LINEAR_VELOCITY = slice(37, 40)
ANGULAR_VELOCITY = slice(40, 43)
JOINT_VELOCITY = slice(43, 61)
BASE_HEIGHT = slice(61, 62)


def quaternion_rotate_inverse_xyzw(
    quaternion: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Rotate vectors by the inverse of Isaac Gym's ``xyzw`` quaternion."""
    if quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("Expected quaternion[...,4] and vector[...,3]")
    quaternion_vector = quaternion[..., :3]
    quaternion_scalar = quaternion[..., 3:4]
    coefficient = 2.0 * torch.square(quaternion_scalar) - 1.0
    return (
        coefficient * vector
        - 2.0
        * quaternion_scalar
        * torch.cross(quaternion_vector, vector, dim=-1)
        + 2.0
        * quaternion_vector
        * torch.sum(quaternion_vector * vector, dim=-1, keepdim=True)
    )


def _body_index(body_names: Sequence[str], side: str, token: str) -> int:
    side_prefix = side.lower() + "_"
    matches = [
        index
        for index, name in enumerate(body_names)
        if str(name).lower().startswith(side_prefix)
        and token in str(name).lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "AMP expected exactly one {} {} body, found {} in {}".format(
                side, token, len(matches), list(body_names)
            )
        )
    return matches[0]


def resolve_amp_key_body_indices(env) -> torch.Tensor:
    """Resolve ``[left foot, right foot, left hand, right hand]``."""
    cached = getattr(env, "_amp_key_body_indices", None)
    if cached is not None:
        return cached
    if not hasattr(env, "body_names"):
        raise RuntimeError("AMP environment does not expose body_names")

    body_names = list(env.body_names)
    if not hasattr(env, "feet_indices") or len(env.feet_indices) != 2:
        raise RuntimeError("AMP requires exactly two foot body indices")
    foot_indices = [int(index) for index in env.feet_indices.tolist()]
    foot_names = [body_names[index].lower() for index in foot_indices]
    if not (
        foot_names[0].startswith("l_") and foot_names[1].startswith("r_")
    ):
        raise RuntimeError(
            "AMP requires [left, right] foot order, received "
            + str(foot_names)
        )

    indices = torch.tensor(
        [
            foot_indices[0],
            foot_indices[1],
            _body_index(body_names, "l", "hand"),
            _body_index(body_names, "r", "hand"),
        ],
        dtype=torch.long,
        device=env.device,
    )
    env._amp_key_body_indices = indices
    return indices


def end_effector_positions_local(env) -> torch.Tensor:
    """Return four end-effector positions in the root/body frame."""
    if not hasattr(env, "rigid_body_states_view"):
        raise RuntimeError("AMP environment lacks rigid_body_states_view")
    indices = resolve_amp_key_body_indices(env)
    world_positions = env.rigid_body_states_view[:, indices, :3]
    root_position = env.root_states[:, :3].unsqueeze(1)
    root_quaternion = env.root_states[:, 3:7].unsqueeze(1).expand(
        -1, world_positions.shape[1], -1
    )
    return quaternion_rotate_inverse_xyzw(
        root_quaternion, world_positions - root_position
    )


def _terrain_height_below_root(env) -> torch.Tensor:
    if hasattr(env, "terrain_h") and env.terrain_h is not None:
        terrain_height = env.terrain_h
    elif hasattr(env, "_sample_terrain_height_xy"):
        terrain_height = env._sample_terrain_height_xy(
            env.root_states[:, :2]
        )
    elif hasattr(env, "measured_heights") and env.measured_heights is not None:
        terrain_height = env.measured_heights.mean(dim=1)
    else:
        terrain_height = torch.zeros(
            env.num_envs,
            dtype=env.root_states.dtype,
            device=env.device,
        )
    return terrain_height.reshape(env.num_envs)


def build_amp_full_frame(env) -> torch.Tensor:
    """Build the 62-value MotionLoaderNing full-frame representation."""
    if int(env.dof_pos.shape[1]) != 18:
        raise RuntimeError(
            "MotionLoaderNing requires 18 DOFs, received {}".format(
                env.dof_pos.shape[1]
            )
        )
    key_positions = end_effector_positions_local(env).reshape(
        env.num_envs, 12
    )
    base_height = (
        env.root_states[:, 2] - _terrain_height_below_root(env)
    ).unsqueeze(1)
    full_frame = torch.cat(
        (
            env.root_states[:, :3],
            env.root_states[:, 3:7],
            env.dof_pos,
            key_positions,
            env.base_lin_vel,
            env.base_ang_vel,
            env.dof_vel,
            base_height,
        ),
        dim=1,
    )
    if full_frame.shape[1] != AMP_FULL_FRAME_DIM:
        raise RuntimeError(
            "AMP full-frame width is {}, expected {}".format(
                full_frame.shape[1], AMP_FULL_FRAME_DIM
            )
        )
    if not torch.isfinite(full_frame).all():
        raise FloatingPointError("AMP full frame contains NaN or Inf")
    return full_frame


def extract_amp_observation(env) -> torch.Tensor:
    """Return the root-invariant 55-value discriminator observation."""
    observation = build_amp_full_frame(env)[:, 7:]
    if observation.shape[1] != AMP_OBSERVATION_DIM:
        raise RuntimeError("Incorrect AMP observation width")
    return observation


def mask_unsupported_amp_features(observation: torch.Tensor) -> torch.Tensor:
    """Zero joints which are intentionally absent from reference motions."""
    if observation.shape[-1] != AMP_OBSERVATION_DIM:
        raise ValueError(
            "AMP observation width must be {}, received {}".format(
                AMP_OBSERVATION_DIM, observation.shape[-1]
            )
        )
    masked = observation.clone()
    masked[..., list(AMP_MASKED_OBSERVATION_INDICES)] = 0.0
    return masked


def mirror_amp_full_frames(
    frames: torch.Tensor, mirror_joints
) -> torch.Tensor:
    """Reflect full frames across the sagittal plane.

    Root rotation is retained because it is not part of the discriminator
    observation.  Joint signs/swaps are delegated to the environment's
    verified N2 action reflection.
    """
    if frames.ndim != 2 or frames.shape[1] != AMP_FULL_FRAME_DIM:
        raise ValueError(
            "Expected AMP full frames [N, {}], received {}".format(
                AMP_FULL_FRAME_DIM, tuple(frames.shape)
            )
        )
    mirrored = frames.clone()
    mirrored[:, 1] *= -1.0
    mirrored[:, JOINT_POSITION] = mirror_joints(
        frames[:, JOINT_POSITION]
    )

    key_positions = frames[:, KEY_POSITION].reshape(-1, 4, 3)
    key_positions = key_positions[:, [1, 0, 3, 2], :].clone()
    key_positions[:, :, 1] *= -1.0
    mirrored[:, KEY_POSITION] = key_positions.reshape(-1, 12)

    mirrored[:, LINEAR_VELOCITY] = frames[:, LINEAR_VELOCITY]
    mirrored[:, 38] *= -1.0
    mirrored[:, ANGULAR_VELOCITY] = frames[:, ANGULAR_VELOCITY]
    mirrored[:, 40] *= -1.0
    mirrored[:, 42] *= -1.0
    mirrored[:, JOINT_VELOCITY] = mirror_joints(
        frames[:, JOINT_VELOCITY]
    )
    return mirrored


def comma_separated_paths(values: Iterable[str]) -> list:
    """Normalize comma-separated path arguments while preserving order."""
    paths = []
    seen = set()
    for value in values:
        for item in str(value).split(","):
            path = item.strip()
            if path and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths
