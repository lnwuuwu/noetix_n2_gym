"""Pure NumPy helpers for the deterministic N2 stair curriculum.

This module intentionally has no Isaac Gym dependency, so geometry and
curriculum dimensions can be validated on a CPU-only development machine.
"""

from typing import Dict, Sequence, Tuple

import numpy as np


def classify_tread_transition(
    valid_landing,
    candidate_tread,
    candidate_foot,
    previous_tread,
    previous_foot,
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
    )
    return (
        advanced,
        alternating_advance,
        repeated_lead,
        same_tread_join,
        skipped_tread,
    )


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
