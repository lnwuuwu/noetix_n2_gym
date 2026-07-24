"""Inference-time left/right policy reflection utilities."""

import math

import torch


def validate_reflection_blend(value):
    """Return a safe reflection blend in ``[0, 0.5]``.

    A blend of zero is the original policy.  A blend of one half evaluates
    ``0.5 * (pi(o) + M_a(pi(M_o(o))))`` and is exactly reflection equivariant.
    Intermediate values provide a conservative, behaviour-preserving probe.
    """
    blend = float(value)
    if not math.isfinite(blend) or not 0.0 <= blend <= 0.5:
        raise ValueError("policy reflection blend must be inside [0.0, 0.5]")
    return blend


def make_reflection_blended_policy(policy, environment, blend):
    """Wrap ``policy`` with a deterministic left/right reflection blend."""
    blend = validate_reflection_blend(blend)
    if blend == 0.0:
        return policy
    if not (
        hasattr(environment, "mirror_observations")
        and hasattr(environment, "mirror_actions")
    ):
        raise ValueError(
            "reflection-blended policy requires observation/action mirrors"
        )

    def blended_policy(observations):
        direct = policy(observations)
        mirrored_observations = environment.mirror_observations(observations)
        mirrored_actions = policy(mirrored_observations)
        reflected_actions = environment.mirror_actions(mirrored_actions)
        return torch.lerp(direct, reflected_actions, blend)

    return blended_policy
