"""Pure NumPy helpers for the deterministic N2 stair curriculum.

This module intentionally has no Isaac Gym dependency, so geometry and
curriculum dimensions can be validated on a CPU-only development machine.
"""

from typing import Dict, Sequence, Tuple

import numpy as np


def stable_landing_mask(stable_contacts, last_stable_contacts):
    """Return one-foot landings that acquire stable opposite-foot support.

    The helper deliberately operates on either NumPy arrays or Torch tensors,
    which keeps the touchdown state machine CPU-testable without importing
    Isaac Gym.  Exactly one foot may become stable on a valid landing.
    """
    new_stable_contact = stable_contacts & ~last_stable_contacts
    one_new_stable_contact = new_stable_contact.sum(-1) == 1
    opposite_was_stable = last_stable_contacts[:, [1, 0]]
    return (
        new_stable_contact
        & opposite_was_stable
        & one_new_stable_contact[..., None]
    )


def _smootherstep01(value):
    """Return a C2 smootherstep after clipping an array/tensor to [0, 1]."""
    value = value.clip(0.0, 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def smooth_swing_trajectory(
    start,
    landing,
    progress,
    arc_height,
    forward_delay=0.15,
    lift_end=0.35,
    descent_start=0.72,
):
    """Interpolate a three-stage C2 toe- and heel-clearing swing.

    The foot first rises with no forward motion, stays at its apex while the
    full 18.5 cm sole crosses the riser, then descends to the next tread. All
    transitions use fifth-order smootherstep, so position, velocity, and
    acceleration remain continuous instead of exciting PD-control chatter.
    """
    if not 0.0 <= forward_delay < lift_end < descent_start < 1.0:
        raise ValueError(
            "Expected 0 <= forward_delay < lift_end < descent_start < 1"
        )
    xy_progress = _smootherstep01(
        (progress - forward_delay) / (1.0 - forward_delay)
    )
    lift_progress = _smootherstep01(progress / lift_end)
    descent_progress = _smootherstep01(
        (progress - descent_start) / (1.0 - descent_start)
    )
    target = start + xy_progress[..., None] * (landing - start)
    target = target.copy() if isinstance(target, np.ndarray) else target.clone()
    candidate_apex = (
        0.5 * (start[..., 2] + landing[..., 2]) + arc_height
    )
    endpoint_high = 0.5 * (
        start[..., 2]
        + landing[..., 2]
        + abs(start[..., 2] - landing[..., 2])
    )
    apex = 0.5 * (
        candidate_apex
        + endpoint_high
        + abs(candidate_apex - endpoint_high)
    )
    target[..., 2] = (
        start[..., 2]
        + lift_progress * (apex - start[..., 2])
        + descent_progress * (landing[..., 2] - apex)
    )
    return target


def retained_swing_support_mask(
    pending_swing,
    new_swing,
    previous_valid,
    opposite_stable_support,
    opposite_true_airborne,
):
    """Keep a support-to-support swing valid until the stance foot lifts.

    The opposite foot must provide stable support when the swing transaction
    starts.  Once started, a temporary failure of the *stable* contact filter
    must not erase the whole transaction: tangential stance-foot motion can
    exceed that filter's slip threshold even though the foot never leaves the
    tread.  The independent force-and-clearance measurement is the physical
    authority after lift-off, so only a genuinely airborne stance foot
    permanently invalidates the transaction.
    """
    opposite_grounded = ~opposite_true_airborne
    started_valid = (
        new_swing & opposite_stable_support & opposite_grounded
    ) | (
        ~new_swing & previous_valid
    )
    return pending_swing & started_valid & opposite_grounded


def true_airborne_mask(
    force_norm,
    ankle_clearance,
    force_threshold,
    clearance_threshold,
):
    """Reject contact dropouts and riser pushes as false lift-off events."""
    return (force_norm < force_threshold) & (
        ankle_clearance > clearance_threshold
    )


def same_tread_support_mask(stable_contacts, tread_indices, num_steps):
    """Identify step-to double support only on intermediate stair treads."""
    both_stable = stable_contacts.all(-1)
    same_tread = tread_indices[:, 0] == tread_indices[:, 1]
    intermediate = (tread_indices[:, 0] > 0) & (
        tread_indices[:, 0] < num_steps
    )
    return both_stable & same_tread & intermediate


def stable_tread_advance_mask(
    stable_measurement,
    geometry_valid,
    landing_pending,
    tread_indices,
    accepted_tread_indices,
):
    """Return stable higher-tread candidates produced by real swings.

    This event is deliberately keyed to the per-foot accepted tread rather
    than to a single contact edge. A sole that first touches a riser boundary
    may become geometrically valid a few frames later and must still advance
    the physical target exactly once. The caller handles simultaneous events
    separately so a two-foot hop can synchronize physical state without being
    rewarded as natural gait.
    """
    return (
        stable_measurement
        & geometry_valid
        & landing_pending
        & (tread_indices > accepted_tread_indices)
    )


def classify_tread_transition(
    valid_landing,
    candidate_tread,
    candidate_foot,
    previous_tread,
    previous_foot,
    previous_joined_tread,
):
    """Classify one landing using NumPy- or Torch-compatible operators.

    The advancing foot must move to exactly the next tread and differ from the
    foot that last advanced. A landing by the other foot on the existing tread
    is a step-to join, not stair-over-stair progress. Inputs may be scalars or
    equally shaped NumPy/Torch arrays; the returned tuple preserves that type.
    """
    advanced = valid_landing & (candidate_tread > previous_tread)
    sequential = candidate_tread == (previous_tread + 1)
    changed_foot = (previous_foot < 0) | (candidate_foot != previous_foot)
    alternating_advance = advanced & sequential & changed_foot
    repeated_lead = advanced & ~changed_foot
    skipped_tread = advanced & ~sequential
    same_tread_join = (
        valid_landing
        & (previous_tread > 0)
        & (candidate_tread == previous_tread)
        & (candidate_foot != previous_foot)
        & (candidate_tread != previous_joined_tread)
    )
    return (
        advanced,
        alternating_advance,
        repeated_lead,
        same_tread_join,
        skipped_tread,
    )


def next_swing_phase_offset(elapsed, gait_frequency, next_swing_foot):
    """Align the observable clock with the requested next swing foot.

    Foot indices are left=0 and right=1.  The configured contact clock assigns
    right swing to phase ``(0.0, 0.5)`` and left swing to ``(0.5, 1.0)``.
    Returning an offset instead of a phase lets the caller retain its normal
    elapsed-time clock while synchronizing it after a physical touchdown.
    NumPy arrays, Torch tensors, and scalar numeric inputs are all supported.
    """
    swing_start_phase = 0.5 * (1 - next_swing_foot)
    return (
        swing_start_phase - elapsed * gait_frequency
    ) % 1.0


def validate_stair_parameters(
    terrain_length: float,
    terrain_width: float,
    horizontal_scale: float,
    vertical_scale: float,
    start_platform_length: float,
    step_width: float,
    step_height: float,
    num_steps: int,
) -> None:
    """Raise ``ValueError`` when a stair flight cannot fit in its terrain tile."""
    positive_values = {
        "terrain_length": terrain_length,
        "terrain_width": terrain_width,
        "horizontal_scale": horizontal_scale,
        "vertical_scale": vertical_scale,
        "start_platform_length": start_platform_length,
        "step_width": step_width,
        "step_height": step_height,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError("{} must be positive, got {}".format(name, value))
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    flight_end = start_platform_length + num_steps * step_width
    if flight_end >= terrain_length:
        raise ValueError(
            "stair flight ends at {:.3f} m but tile length is {:.3f} m".format(
                flight_end, terrain_length
            )
        )


def build_directional_stairs(
    terrain_length: float,
    terrain_width: float,
    horizontal_scale: float,
    vertical_scale: float,
    start_platform_length: float,
    step_width: float,
    step_height: float,
    num_steps: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build one staircase that rises strictly along local +X.

    Returns an ``int16`` height field in Isaac Gym height units and useful
    metric metadata. The low platform occupies ``[0, start_platform_length)``;
    after ``num_steps`` treads the remainder of the tile is a flat top platform.
    """
    validate_stair_parameters(
        terrain_length,
        terrain_width,
        horizontal_scale,
        vertical_scale,
        start_platform_length,
        step_width,
        step_height,
        num_steps,
    )
    length_pixels = int(round(terrain_length / horizontal_scale))
    width_pixels = int(round(terrain_width / horizontal_scale))
    start_pixel = int(round(start_platform_length / horizontal_scale))
    tread_pixels = max(1, int(round(step_width / horizontal_scale)))
    step_units = max(1, int(round(step_height / vertical_scale)))

    field = np.zeros((length_pixels, width_pixels), dtype=np.int16)
    for step_index in range(num_steps):
        tread_start = start_pixel + step_index * tread_pixels
        tread_end = min(start_pixel + (step_index + 1) * tread_pixels, length_pixels)
        field[tread_start:tread_end, :] = (step_index + 1) * step_units

    top_start_pixel = min(start_pixel + num_steps * tread_pixels, length_pixels)
    field[top_start_pixel:, :] = num_steps * step_units

    realized_step_height = step_units * vertical_scale
    metadata = {
        "start_x": start_pixel * horizontal_scale,
        "top_start_x": top_start_pixel * horizontal_scale,
        "success_x": min(
            terrain_length - horizontal_scale,
            top_start_pixel * horizontal_scale + 0.20,
        ),
        "top_height": num_steps * realized_step_height,
        "step_height": realized_step_height,
        "step_width": tread_pixels * horizontal_scale,
    }
    return field, metadata


def terrain_height_at_x(
    x_positions: np.ndarray,
    start_x: float,
    step_width: float,
    step_height: float,
    num_steps: int,
) -> np.ndarray:
    """Analytic height lookup used by the optional MuJoCo observation adapter."""
    x_positions = np.asarray(x_positions, dtype=np.float64)
    relative = x_positions - start_x
    completed_risers = np.where(
        relative < 0.0,
        0,
        np.floor(relative / step_width).astype(np.int64) + 1,
    )
    completed_risers = np.clip(completed_risers, 0, num_steps)
    return completed_risers.astype(np.float64) * step_height


def select_height_indices(
    measured_points_x: Sequence[float],
    measured_points_y: Sequence[float],
    actor_points_x: Sequence[float],
    actor_points_y: Sequence[float],
    tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Return flattened meshgrid indices for the Actor's compact height scan."""
    indices = []
    for actor_x in actor_points_x:
        x_matches = [
            index for index, value in enumerate(measured_points_x)
            if abs(value - actor_x) <= tolerance
        ]
        if len(x_matches) != 1:
            raise ValueError("Actor X point {} is not unique in measured_points_x".format(actor_x))
        for actor_y in actor_points_y:
            y_matches = [
                index for index, value in enumerate(measured_points_y)
                if abs(value - actor_y) <= tolerance
            ]
            if len(y_matches) != 1:
                raise ValueError("Actor Y point {} is not unique in measured_points_y".format(actor_y))
            indices.append(x_matches[0] * len(measured_points_y) + y_matches[0])
    return np.asarray(indices, dtype=np.int64)
