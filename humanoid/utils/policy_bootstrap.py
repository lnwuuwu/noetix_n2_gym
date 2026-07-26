"""Safe Actor-only bootstrapping across the two N2 stair observation layouts.

The approved ``n2_stairs_walk`` Actor consumes five 82-value frames, while
``n2_faststair`` consumes five 115-value frames.  Both layouts start with the
same 70 deployable proprioceptive/navigation values.  The remaining values are
terrain samples: 12 in the legacy layout and 45 in the wider FastStair layout.

This module transfers the complete legacy Actor, maps every legacy terrain
sample to the *same physical coordinate* in the FastStair map, and leaves
genuinely new map inputs at zero.  The Critic is deliberately not transferred
because its observation layout changed.  As a result FastStair starts with a
proven climbing behaviour instead of a random policy, while still being able
to learn from its larger map.
"""

from __future__ import annotations

import os

import torch


LEGACY_FRAME_SIZE = 82
FASTSTAIR_FRAME_SIZE = 115
SHARED_PROPRIO_SIZE = 70
FRAME_STACK = 5

LEGACY_TERRAIN_X = (0.25, 0.45, 0.65, 0.85)
LEGACY_TERRAIN_Y = (-0.24, 0.0, 0.24)
FASTSTAIR_TERRAIN_X = (
    0.15,
    0.25,
    0.45,
    0.65,
    0.85,
    1.05,
    1.15,
    1.20,
    1.35,
)
FASTSTAIR_TERRAIN_Y = (-0.24, -0.12, 0.0, 0.12, 0.24)


def _grid_points(x_values, y_values):
    return tuple(
        (float(x_value), float(y_value))
        for x_value in x_values
        for y_value in y_values
    )


def nearest_terrain_mapping():
    """Return the exact legacy-index to FastStair-index coordinate mapping.

    The historical public name is retained for checkpoint/tool compatibility.
    A nearest-neighbour approximation is intentionally no longer accepted:
    across a stair edge, a displacement of only a few centimetres can change a
    height observation by a full riser and invalidate Actor parity.
    """
    source = _grid_points(LEGACY_TERRAIN_X, LEGACY_TERRAIN_Y)
    target = _grid_points(FASTSTAIR_TERRAIN_X, FASTSTAIR_TERRAIN_Y)
    mapping = []
    for source_point in source:
        matches = [
            index
            for index, target_point in enumerate(target)
            if all(
                abs(source_value - target_value) <= 1.0e-9
                for source_value, target_value in zip(
                    source_point, target_point
                )
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Legacy terrain coordinate {} has {} exact FastStair "
                "matches; Actor bootstrap requires one".format(
                    source_point, len(matches)
                )
            )
        mapping.append(matches[0])
    if len(set(mapping)) != len(mapping):
        raise RuntimeError(
            "Legacy terrain samples do not map uniquely into FastStair grid"
        )
    return tuple(mapping)


def _copy_tensor(target, source, name):
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            "{} shape mismatch: source {} vs target {}".format(
                name, tuple(source.shape), tuple(target.shape)
            )
        )
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def bootstrap_faststair_actor(policy, checkpoint_path):
    """Load a legacy stair Actor into a fresh FastStair policy.

    Args:
        policy: Fresh FastStair ``ActorCritic`` instance.
        checkpoint_path: Approved legacy ``model_N.pt`` checkpoint.

    Returns:
        A small audit dictionary suitable for a launcher log.
    """
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError(
            "{} has no model_state_dict".format(checkpoint_path)
        )
    target_state = policy.state_dict()
    source_first = source_state.get("actor.0.weight")
    target_first = target_state.get("actor.0.weight")
    if source_first is None or target_first is None:
        raise ValueError("Source and target policies require actor.0.weight")
    expected_source_width = LEGACY_FRAME_SIZE * FRAME_STACK
    expected_target_width = FASTSTAIR_FRAME_SIZE * FRAME_STACK
    if source_first.ndim != 2 or source_first.shape[1] != expected_source_width:
        raise ValueError(
            "Legacy Actor input must be {}, received {}".format(
                expected_source_width, tuple(source_first.shape)
            )
        )
    if target_first.ndim != 2 or target_first.shape[1] != expected_target_width:
        raise ValueError(
            "FastStair Actor input must be {}, received {}".format(
                expected_target_width, tuple(target_first.shape)
            )
        )
    if source_first.shape[0] != target_first.shape[0]:
        raise ValueError(
            "Actor first-layer output mismatch: {} vs {}".format(
                source_first.shape[0], target_first.shape[0]
            )
        )

    terrain_mapping = nearest_terrain_mapping()
    actor_keys = sorted(
        key for key in source_state if key.startswith("actor.")
    )
    if not actor_keys:
        raise ValueError("Legacy checkpoint contains no Actor parameters")

    with torch.no_grad():
        target_first.zero_()
        source_first = source_first.to(
            device=target_first.device, dtype=target_first.dtype
        )
        for frame in range(FRAME_STACK):
            source_offset = frame * LEGACY_FRAME_SIZE
            target_offset = frame * FASTSTAIR_FRAME_SIZE
            target_first[
                :,
                target_offset : target_offset + SHARED_PROPRIO_SIZE,
            ].copy_(
                source_first[
                    :,
                    source_offset : source_offset + SHARED_PROPRIO_SIZE,
                ]
            )
            for source_index, target_index in enumerate(terrain_mapping):
                target_first[
                    :,
                    target_offset + SHARED_PROPRIO_SIZE + target_index,
                ].copy_(
                    source_first[
                        :,
                        source_offset + SHARED_PROPRIO_SIZE + source_index,
                    ]
                )

        for key in actor_keys:
            if key == "actor.0.weight":
                continue
            if key not in target_state:
                raise ValueError(
                    "FastStair Actor is missing source parameter {}".format(
                        key
                    )
                )
            _copy_tensor(target_state[key], source_state[key], key)

        noise_key = None
        for candidate in ("std", "log_std"):
            if candidate in source_state and candidate in target_state:
                _copy_tensor(
                    target_state[candidate],
                    source_state[candidate],
                    candidate,
                )
                noise_key = candidate
                break

    return {
        "checkpoint": checkpoint_path,
        "source_iteration": int(checkpoint.get("iter", 0)),
        "source_actor_obs": expected_source_width,
        "target_actor_obs": expected_target_width,
        "shared_per_frame": SHARED_PROPRIO_SIZE,
        "mapped_terrain_per_frame": len(terrain_mapping),
        "new_terrain_per_frame": (
            len(FASTSTAIR_TERRAIN_X) * len(FASTSTAIR_TERRAIN_Y)
            - len(terrain_mapping)
        ),
        "noise_parameter": noise_key,
    }
