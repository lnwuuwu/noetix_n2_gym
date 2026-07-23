"""Pure phase-driven action references for native MuJoCo stair training.

The reference is deliberately expressed in the policy's action coordinates.
It can therefore be blended into the training plant without changing the
Actor observation or exported policy.  Deterministic model selection remains
unassisted; this module only supplies a temporary exploration scaffold and an
action-imitation target.
"""

import math

import numpy as np


def _smoothstep01(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def phase_swing_action_reference(
    phase,
    step_height,
    action_scale,
    num_actions,
    sagittal_joint_indices,
    guidance_cfg,
):
    """Return an alternating swing-leg reference in policy action units.

    Phase ``[0, 0.5)`` schedules the right leg and ``[0.5, 1)`` schedules the
    left leg, matching the contact convention used by the training and
    deployment environments.  A negative hip-pitch offset advances the N2
    thigh, positive knee pitch provides clearance, and the ankle cancels their
    sum to keep the foot approximately level.

    Returns ``(reference, mask, swing_foot, progress, phase_weight)``.  The
    reference is zero outside the three sagittal joints of the scheduled leg;
    callers should blend only entries selected by ``mask``.
    """

    phase = float(phase) % 1.0
    step_height = float(step_height)
    action_scale = float(action_scale)
    if action_scale <= 0.0:
        raise ValueError("action_scale must be positive")
    if int(num_actions) <= 0:
        raise ValueError("num_actions must be positive")

    indices = np.asarray(sagittal_joint_indices, dtype=np.int64)
    if indices.shape != (2, 3):
        raise ValueError(
            "sagittal_joint_indices must have shape (2, 3)"
        )
    if np.any(indices < 0) or np.any(indices >= int(num_actions)):
        raise ValueError("sagittal joint index is outside the action")

    # The observable clock agrees with _desired_contacts(): right swings in
    # the first half-cycle, then left.  Keeping this target phase-only avoids
    # introducing hidden contact-history state into the Actor's objective.
    swing_foot = 1 if phase < 0.5 else 0
    progress = float((2.0 * phase) % 1.0)
    motion_start = float(guidance_cfg.get("motion_start_phase", 0.08))
    if not 0.0 <= motion_start < 1.0:
        raise ValueError("motion_start_phase must be in [0, 1)")
    motion_progress = float(
        np.clip(
            (progress - motion_start) / max(1.0 - motion_start, 1.0e-6),
            0.0,
            1.0,
        )
    )
    forward_profile = _smoothstep01(motion_progress)
    clearance_profile = math.sin(math.pi * motion_progress) ** 2
    phase_weight = _smoothstep01(
        motion_progress
        / max(
            float(guidance_cfg.get("blend_ramp_fraction", 0.20)),
            1.0e-6,
        )
    )

    reference_height = float(
        guidance_cfg.get("reference_step_height_m", 0.02)
    )
    height_excess = max(step_height - reference_height, 0.0)
    hip_landing = (
        float(guidance_cfg["hip_landing_offset_rad"])
        + float(guidance_cfg["hip_landing_height_gain"])
        * height_excess
    )
    knee_clearance = (
        float(guidance_cfg["knee_clearance_offset_rad"])
        + float(guidance_cfg["knee_clearance_height_gain"])
        * height_excess
    )
    knee_landing = (
        float(guidance_cfg["knee_landing_offset_rad"])
        + float(guidance_cfg["knee_landing_height_gain"])
        * height_excess
    )

    hip_delta = -hip_landing * forward_profile
    knee_delta = (
        knee_landing * forward_profile
        + knee_clearance * clearance_profile
    )
    ankle_delta = -(hip_delta + knee_delta)

    reference = np.zeros(int(num_actions), dtype=np.float32)
    mask = np.zeros(int(num_actions), dtype=bool)
    selected = indices[swing_foot]
    reference[selected] = (
        np.asarray(
            [hip_delta, knee_delta, ankle_delta], dtype=np.float32
        )
        / action_scale
    )
    mask[selected] = True
    return reference, mask, swing_foot, progress, phase_weight


def blend_swing_action(
    policy_action,
    reference,
    mask,
    assistance_scale,
    phase_weight,
):
    """Blend a bounded reference into only the scheduled swing-leg joints."""

    policy_action = np.asarray(policy_action)
    reference = np.asarray(reference)
    mask = np.asarray(mask, dtype=bool)
    if (
        policy_action.shape != reference.shape
        or policy_action.shape != mask.shape
    ):
        raise ValueError("policy action, reference, and mask must align")
    scale = float(np.clip(assistance_scale, 0.0, 1.0))
    scale *= float(np.clip(phase_weight, 0.0, 1.0))
    blended = policy_action.copy()
    blended[mask] = (
        (1.0 - scale) * policy_action[mask]
        + scale * reference[mask]
    )
    return blended, scale
