"""Deterministic Isaac-Gym parkour terrain generation for MuJoCo.

This module deliberately has no Isaac Gym dependency.  It mirrors the
``ParkourTerrain`` curriculum used by ``humanoid/scripts/play.py``:

* NumPy's legacy MT19937 stream is seeded with 5 by default.
* The complete 8 row x 10 column curriculum is generated in column-major
  order, even though only row 3 and five representative columns are retained.
* Geometry is quantised at 0.1 m horizontally and 0.005 m vertically.
* ``random_uniform_terrain`` roughness is reproduced with a pure-NumPy
  bilinear resize.

The paper-compatible stair run range remains the default.  Evaluation configs
may request a wider run range; the same generated :class:`Campaign` then owns
the collision height field, height observations, goals and stage boundaries,
so those views can never drift apart.

The retained tiles form one continuous +x course:

    up stairs -> down stairs -> hurdle -> flat -> stepping stones

There is no teleport wall between stages.  Long unused tail platforms are
cropped to 0.8 m after each final goal.  Only the two ends of the whole course
and its two lateral edges keep the 0.5 m parkour pad.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np


HORIZONTAL_SCALE = 0.1
VERTICAL_SCALE = 0.005
ARENA_LENGTH = 18.0
ARENA_WIDTH = 4.0
NUM_ROWS = 8
NUM_COLS = 10
NUM_GOALS = 10

KIND_UP_STAIRS = "up_stairs"
KIND_DOWN_STAIRS = "down_stairs"
KIND_HURDLE = "hurdle"
KIND_FLAT = "flat"
KIND_STEPPING_STONES = "stepping_stones"

STAGE_COLUMNS = (0, 3, 6, 8, 9)
STAGE_KINDS = (
    KIND_UP_STAIRS,
    KIND_DOWN_STAIRS,
    KIND_HURDLE,
    KIND_FLAT,
    KIND_STEPPING_STONES,
)
STAGE_LABELS = ("上台阶", "下台阶", "跨栏", "平地路点", "踏石")

_PROPORTIONS = np.cumsum([0.3, 0.3, 0.2, 0.1, 0.1])
_PAD_WIDTH = 0.1
_PAD_HEIGHT = 0.5
_PLATFORM_LENGTH = 2.5
DEFAULT_STEP_X_RANGE = (0.2, 0.4)


@dataclass(frozen=True)
class Stage:
    """One contiguous part of a :class:`Campaign`.

    ``height_field_raw`` is local to the stage and indexed ``[x, y]``.
    ``goals`` is always a ``(10, 3)`` array in campaign/world coordinates.
    """

    index: int
    row: int
    col: int
    kind: str
    label: str
    difficulty: float
    origin_x: float
    height_field_raw: np.ndarray
    goals: np.ndarray
    height_offset: float = 0.0

    @property
    def end_x(self) -> float:
        return self.origin_x + self.height_field_raw.shape[0] * HORIZONTAL_SCALE


@dataclass(frozen=True)
class Campaign:
    """A five-stage, continuous parkour height field."""

    stages: Tuple[Stage, ...]
    height_field_raw: np.ndarray
    horizontal_scale: float
    vertical_scale: float
    arena_length: float
    arena_width: float
    seed: int
    row: int
    add_roughness: bool
    center_flat_goals: bool = False
    step_x_range: Tuple[float, float] = DEFAULT_STEP_X_RANGE

    @property
    def length(self) -> float:
        return self.height_field_raw.shape[0] * self.horizontal_scale

    @property
    def width(self) -> float:
        return self.height_field_raw.shape[1] * self.horizontal_scale

    @property
    def heights(self) -> np.ndarray:
        """Height field in metres, indexed ``[x, y]``."""

        return self.height_field_raw.astype(np.float64) * self.vertical_scale

    @property
    def goals(self) -> np.ndarray:
        """All stage goals concatenated in traversal order, shape ``(50, 3)``."""

        if not self.stages:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate([stage.goals for stage in self.stages], axis=0)

    @property
    def goals_by_stage(self) -> np.ndarray:
        """Goals grouped by stage, shape ``(5, 10, 3)``."""

        if not self.stages:
            return np.empty((0, NUM_GOALS, 3), dtype=np.float64)
        return np.stack([stage.goals for stage in self.stages], axis=0)

    def sample_heights(self, points_xy: np.ndarray) -> np.ndarray:
        """Sample terrain exactly like Isaac Gym's ``_get_heights``.

        Coordinates are converted to integer ``px/py`` indices by truncating
        toward zero, clipped to leave room for neighbours, then the minimum of
        ``(px, py)``, ``(px + 1, py)`` and ``(px, py + 1)`` is returned.
        The y origin of the height field is ``-width / 2``.
        """

        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError("points_xy must have shape (..., 2)")
        shape = points.shape[:-1]
        flat = points.reshape(-1, 2)
        px = (flat[:, 0] / self.horizontal_scale).astype(np.int64)
        py = ((flat[:, 1] + self.width / 2.0) /
              self.horizontal_scale).astype(np.int64)
        px = np.clip(px, 0, self.height_field_raw.shape[0] - 2)
        py = np.clip(py, 0, self.height_field_raw.shape[1] - 2)
        h0 = self.height_field_raw[px, py]
        hx = self.height_field_raw[px + 1, py]
        hy = self.height_field_raw[px, py + 1]
        raw = np.minimum(np.minimum(h0, hx), hy)
        return (raw.astype(np.float64) * self.vertical_scale).reshape(shape)


def sample_heights(campaign: Campaign, points_xy: np.ndarray) -> np.ndarray:
    """Functional form of :meth:`Campaign.sample_heights`."""

    return campaign.sample_heights(points_xy)


def _config_value(
        config: Optional[Mapping[str, Any]],
        names: Sequence[str],
        default: Any,
) -> Any:
    if config is None:
        return default
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping or None")
    campaign_cfg = config.get("campaign")
    sources = ((campaign_cfg, config)
               if isinstance(campaign_cfg, Mapping) else (config,))
    for source in sources:
        for name in names:
            if name in source:
                return source[name]
    return default


def _validated_step_x_range(values: Sequence[float]) -> Tuple[float, float]:
    """Validate a half-open stair run range in metres.

    One MuJoCo height-field node is consumed by the interpolated riser.  The
    slow-stable course therefore uses a range shifted by one 0.1 m cell while
    the default remains paper-compatible.
    """

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "step_x_range must contain exactly two finite numbers") from exc
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(
            "step_x_range must contain exactly two finite numbers")
    low, high = (float(value) for value in array)
    if low <= 0.0 or high <= low:
        raise ValueError("step_x_range must satisfy 0 < low < high")

    min_cells = int(np.nextafter(low, np.inf) / HORIZONTAL_SCALE)
    max_cells = int(np.nextafter(high, -np.inf) / HORIZONTAL_SCALE)
    if min_cells < 1:
        raise ValueError(
            "step_x_range is too short for the 0.1 m terrain grid")
    platform_cells = int(_PLATFORM_LENGTH / HORIZONTAL_SCALE)
    goal_margin_cells = int(0.5 / HORIZONTAL_SCALE)
    arena_cells = int(ARENA_LENGTH / HORIZONTAL_SCALE)
    if platform_cells + 18 * max_cells + goal_margin_cells >= arena_cells:
        raise ValueError("step_x_range is too long for the 18 m stair tile")
    return low, high


def _step_goal_indices(num_steps: int, num_goals: int, power: float = 2.0) -> np.ndarray:
    num_steps = max(int(num_steps), int(num_goals))
    k = np.arange(1, num_goals + 1, dtype=np.float64) / num_goals
    indices = np.maximum(
        np.round(num_steps * k ** power).astype(np.int64), 1)
    for i in range(1, num_goals):
        indices[i] = max(indices[i], indices[i - 1] + 1)
    indices[-1] = num_steps
    for i in range(num_goals - 2, -1, -1):
        indices[i] = min(indices[i], indices[i + 1] - 1)
    return indices


def _pad_edges(raw: np.ndarray) -> None:
    pad_width = max(1, int(_PAD_WIDTH / HORIZONTAL_SCALE))
    pad_height = int(_PAD_HEIGHT / VERTICAL_SCALE)
    raw[:pad_width, :] = pad_height
    raw[-pad_width:, :] = pad_height
    raw[:, :pad_width] = pad_height
    raw[:, -pad_width:] = pad_height


def _parkour_step(
        rng: np.random.RandomState,
        step_height: float,
        outside_margin: Optional[float] = None,
        step_x_range: Sequence[float] = DEFAULT_STEP_X_RANGE,
) -> Tuple[np.ndarray, np.ndarray]:
    nx = int(ARENA_LENGTH / HORIZONTAL_SCALE)
    ny = int(ARENA_WIDTH / HORIZONTAL_SCALE)
    mid_y = ny // 2
    raw = np.zeros((nx, ny), dtype=np.int16)
    lane = np.zeros((nx, ny), dtype=bool)
    goals = np.zeros((NUM_GOALS, 2), dtype=np.float64)

    platform = int(_PLATFORM_LENGTH / HORIZONTAL_SCALE)
    goals[0] = [(platform * 0.5) * HORIZONTAL_SCALE,
                mid_y * HORIZONTAL_SCALE]
    lane[:platform, :] = True
    step_raw = int(step_height / VERTICAL_SCALE)
    num_stones = NUM_GOALS - 2
    num_steps = 18
    goal_at = _step_goal_indices(num_steps, num_stones, 2.0)
    goal_slots = {int(step): index for index, step in enumerate(goal_at)}

    cursor = platform
    current_height = 0
    last_center = goals[0].copy()
    for step in range(1, num_steps + 1):
        run = int(rng.uniform(*step_x_range) / HORIZONTAL_SCALE)
        random_y = int(rng.uniform(-0.15, 0.15) / HORIZONTAL_SCALE)
        half_width = int(rng.uniform(0.7, 0.8) / HORIZONTAL_SCALE)
        x0, x1 = cursor, min(cursor + run, nx)
        if x0 >= nx:
            break
        current_height += step_raw
        y0 = max(0, mid_y + random_y - half_width)
        y1 = min(ny, mid_y + random_y + half_width)
        raw[x0:x1, y0:y1] = current_height
        lane[x0:x1, y0:y1] = True
        last_center = np.array(
            [((x0 + x1) * 0.5) * HORIZONTAL_SCALE,
             (mid_y + random_y) * HORIZONTAL_SCALE])
        if step in goal_slots:
            goals[goal_slots[step] + 1] = last_center
        cursor = x1

    for index in range(num_stones):
        if not goals[index + 1].any():
            goals[index + 1] = last_center

    half_width = int(np.mean((0.7, 0.8)) / HORIZONTAL_SCALE)
    if cursor < nx:
        raw[cursor:, mid_y - half_width:mid_y + half_width] = current_height
        lane[cursor:, mid_y - half_width:mid_y + half_width] = True
    goals[-1] = [
        min(cursor + int(0.5 / HORIZONTAL_SCALE), nx - 1) *
        HORIZONTAL_SCALE,
        mid_y * HORIZONTAL_SCALE,
    ]

    if outside_margin is not None:
        sentinel = np.iinfo(np.int32).min
        masked = np.where(lane, raw.astype(np.int32), sentinel)
        profile = masked.max(axis=1)
        valid = profile > sentinel
        if valid.any():
            profile = np.interp(
                np.arange(nx), np.flatnonzero(valid), profile[valid])
            outside = (
                profile - outside_margin / VERTICAL_SCALE).astype(np.int32)
            raw[:] = np.where(lane, raw, outside[:, None]).astype(np.int16)

    _pad_edges(raw)
    return raw, goals


def _parkour_hurdle(
        rng: np.random.RandomState,
        difficulty: float,
        flat: bool,
        center_flat_goals: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    nx = int(ARENA_LENGTH / HORIZONTAL_SCALE)
    ny = int(ARENA_WIDTH / HORIZONTAL_SCALE)
    mid_y = ny // 2
    raw = np.zeros((nx, ny), dtype=np.int16)
    goals = np.zeros((NUM_GOALS, 2), dtype=np.float64)

    platform = int(_PLATFORM_LENGTH / HORIZONTAL_SCALE)
    stone_length = max(1, int(0.3 / HORIZONTAL_SCALE))
    half_width = int(rng.uniform(0.4, 0.8) / HORIZONTAL_SCALE)
    top = 0.10 + difficulty * (0.30 - 0.10)
    low = max(0.10 * 0.5, top - 0.05)
    low_raw = int(low / VERTICAL_SCALE)
    high_raw = int(top / VERTICAL_SCALE)

    goals[0] = [max(platform - 1, 0) * HORIZONTAL_SCALE,
                mid_y * HORIZONTAL_SCALE]
    cursor = platform
    for index in range(NUM_GOALS - 2):
        random_x = int(rng.uniform(1.2, 1.8) / HORIZONTAL_SCALE)
        random_y = int(rng.uniform(-0.4, 0.4) / HORIZONTAL_SCALE)
        cursor = min(cursor + random_x, nx - 1)
        if not flat:
            x0 = max(0, cursor - stone_length // 2)
            x1 = min(nx, cursor + stone_length // 2 + 1)
            center_y = mid_y + random_y
            raw[
                x0:x1,
                max(0, center_y - half_width):
                min(ny, center_y + half_width),
            ] = rng.randint(low_raw, max(low_raw + 1, high_raw))
        # Always draw random_y to preserve the complete 8x10 RNG stream.  The
        # straight-flat option changes only the goal location, never later
        # terrain geometry or roughness samples.
        goal_y = mid_y if flat and center_flat_goals else mid_y + random_y
        goals[index + 1] = [
            max(cursor - random_x // 2, 0) * HORIZONTAL_SCALE,
            goal_y * HORIZONTAL_SCALE,
        ]

    final = min(cursor + int(np.mean((1.2, 1.8)) /
                                 HORIZONTAL_SCALE), nx - 1)
    goals[-1] = [final * HORIZONTAL_SCALE, mid_y * HORIZONTAL_SCALE]
    _pad_edges(raw)
    return raw, goals


def _parkour_stones(
        rng: np.random.RandomState,
        difficulty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nx = int(ARENA_LENGTH / HORIZONTAL_SCALE)
    ny = int(ARENA_WIDTH / HORIZONTAL_SCALE)
    mid_y = ny // 2
    pit_raw = int(0.5 / VERTICAL_SCALE)
    raw = np.full((nx, ny), -pit_raw, dtype=np.int16)
    goals = np.zeros((NUM_GOALS, 2), dtype=np.float64)

    platform = int(_PLATFORM_LENGTH / HORIZONTAL_SCALE)
    stone_length_m = 0.9 + difficulty * (0.45 - 0.9)
    stone_width_m = 1.0 + difficulty * (0.5 - 1.0)
    stone_length = max(
        1, int(round(stone_length_m / HORIZONTAL_SCALE)))
    stone_width = max(
        2, int(round(stone_width_m / HORIZONTAL_SCALE)))
    raw[:platform, :] = 0
    goals[0] = [(platform * 0.5) * HORIZONTAL_SCALE,
                mid_y * HORIZONTAL_SCALE]

    gap_high = 0.05 + difficulty * (0.20 - 0.05)
    cursor = platform
    side = rng.randint(0, 2)
    for index in range(NUM_GOALS - 2):
        cursor += int(rng.uniform(0.05, gap_high) / HORIZONTAL_SCALE)
        x0, x1 = min(cursor, nx - 1), min(cursor + stone_length, nx)
        center_y = mid_y + (1 if side else -1) * int(
            rng.uniform(0.10, 0.18) / HORIZONTAL_SCALE)
        y0 = max(0, center_y - stone_width // 2)
        raw[x0:x1, y0:min(ny, y0 + stone_width)] = 0
        goals[index + 1] = [
            ((x0 + x1) * 0.5) * HORIZONTAL_SCALE,
            center_y * HORIZONTAL_SCALE,
        ]
        cursor = x1
        side = 1 - side

    if cursor < nx:
        raw[cursor:, :] = 0
    goals[-1] = [
        min(cursor + int(0.5 / HORIZONTAL_SCALE), nx - 1) *
        HORIZONTAL_SCALE,
        mid_y * HORIZONTAL_SCALE,
    ]
    _pad_edges(raw)
    return raw, goals


def _bilinear_resize(values: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Endpoint-aligned bilinear interpolation, implemented only with NumPy."""

    out_x, out_y = shape
    src_x, src_y = values.shape
    x = np.linspace(0.0, src_x - 1.0, out_x)
    y = np.linspace(0.0, src_y - 1.0, out_y)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, src_x - 1)
    y1 = np.minimum(y0 + 1, src_y - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[None, :]
    v00 = values[x0[:, None], y0[None, :]]
    v10 = values[x1[:, None], y0[None, :]]
    v01 = values[x0[:, None], y1[None, :]]
    v11 = values[x1[:, None], y1[None, :]]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )


def _add_random_roughness(
        raw: np.ndarray,
        rng: np.random.RandomState,
) -> None:
    """Mirror ``add_roughness``/``random_uniform_terrain`` RNG and resize."""

    magnitude = rng.uniform(0.01, 0.03)
    min_height = int(-magnitude / VERTICAL_SCALE)
    max_height = int(magnitude / VERTICAL_SCALE)
    step = int(0.005 / VERTICAL_SCALE)
    choices = np.arange(
        min_height, max_height + step, step, dtype=np.int16)
    downsampled_scale = 0.075
    coarse_shape = (
        int(raw.shape[0] * HORIZONTAL_SCALE / downsampled_scale),
        int(raw.shape[1] * HORIZONTAL_SCALE / downsampled_scale),
    )
    coarse = rng.choice(choices, size=coarse_shape)
    noise = np.rint(_bilinear_resize(coarse, raw.shape)).astype(np.int16)
    raw[:] = (raw.astype(np.int32) + noise.astype(np.int32)).astype(np.int16)


def _make_tile(
        rng: np.random.RandomState,
        choice: float,
        difficulty: float,
        center_flat_goals: bool = False,
        step_x_range: Sequence[float] = DEFAULT_STEP_X_RANGE,
) -> Tuple[np.ndarray, np.ndarray]:
    step_height = 0.05 + difficulty * (0.20 - 0.05)
    if choice < _PROPORTIONS[0]:
        return _parkour_step(
            rng, step_height, step_x_range=step_x_range)
    if choice < _PROPORTIONS[1]:
        return _parkour_step(
            rng, -step_height, outside_margin=0.3,
            step_x_range=step_x_range)
    if choice < _PROPORTIONS[3]:
        return _parkour_hurdle(
            rng, difficulty, flat=choice >= _PROPORTIONS[2],
            center_flat_goals=center_flat_goals)
    return _parkour_stones(rng, difficulty)


def _remove_internal_x_pads(
        fields: Sequence[np.ndarray],
) -> None:
    pad = max(1, int(_PAD_WIDTH / HORIZONTAL_SCALE))
    last_index = len(fields) - 1
    for index, raw in enumerate(fields):
        if index != 0:
            raw[:pad, :] = raw[pad:pad + 1, :]
        if index != last_index:
            raw[-pad:, :] = raw[-pad - 1:-pad, :]


def _crop_after_final_goal(
        raw: np.ndarray,
        local_goals: np.ndarray,
        margin: float = 0.8,
) -> np.ndarray:
    """Drop an 18 m tile's unused tail while keeping reset/start platform."""

    end_metres = float(local_goals[-1, 0]) + float(margin)
    # The tiny epsilon prevents a binary representation such as
    # 8.500000000000002 from allocating one unwanted extra cell.
    end_index = int(np.ceil(
        (end_metres - 1e-10) / HORIZONTAL_SCALE))
    end_index = int(np.clip(end_index, 2, raw.shape[0]))
    return raw[:end_index, :].copy()


def _goals_world(
        local_xy: np.ndarray,
        raw: np.ndarray,
        origin_x: float,
) -> np.ndarray:
    px = np.clip(
        (local_xy[:, 0] / HORIZONTAL_SCALE).astype(np.int64),
        0,
        raw.shape[0] - 1,
    )
    py = np.clip(
        (local_xy[:, 1] / HORIZONTAL_SCALE).astype(np.int64),
        0,
        raw.shape[1] - 1,
    )
    goals = np.empty((local_xy.shape[0], 3), dtype=np.float64)
    goals[:, 0] = local_xy[:, 0] + origin_x
    goals[:, 1] = local_xy[:, 1] - ARENA_WIDTH / 2.0
    goals[:, 2] = raw[px, py].astype(np.float64) * VERTICAL_SCALE
    return goals


def build_campaign(
        config: Optional[Mapping[str, Any]] = None,
        *,
        row: Optional[int] = None,
        seed: Optional[int] = None,
        add_roughness: Optional[bool] = None,
        center_flat_goals: Optional[bool] = None,
        step_x_range: Optional[Sequence[float]] = None,
) -> Campaign:
    """Build the deterministic five-stage MuJoCo campaign.

    ``config`` may be the full YAML mapping or a nested ``campaign`` mapping.
    Recognised keys are ``campaign_row``/``row``,
    ``campaign_seed``/``seed`` and
    ``campaign_add_roughness``/``add_roughness`` and
    ``campaign_center_flat_goals``/``center_flat_goals`` and
    ``campaign_step_x_range``/``step_x_range``. Explicit keyword arguments
    take precedence.
    """

    if row is None:
        row = int(_config_value(
            config, ("campaign_row", "terrain_row", "row"), 3))
    if seed is None:
        seed = int(_config_value(
            config, ("campaign_seed", "terrain_seed", "seed"), 5))
    if add_roughness is None:
        add_roughness = bool(_config_value(
            config,
            ("campaign_add_roughness", "terrain_add_roughness",
             "add_roughness"),
            True,
        ))
    if center_flat_goals is None:
        center_flat_goals = bool(_config_value(
            config,
            ("campaign_center_flat_goals", "terrain_center_flat_goals",
             "center_flat_goals"),
            False,
        ))
    if step_x_range is None:
        step_x_range = _config_value(
            config,
            ("campaign_step_x_range", "terrain_step_x_range",
             "step_x_range"),
            DEFAULT_STEP_X_RANGE,
        )
    step_x_range = _validated_step_x_range(step_x_range)
    if not 0 <= int(row) < NUM_ROWS:
        raise ValueError("row must be in [0, 7]")
    row = int(row)
    seed = int(seed)
    difficulty = row / NUM_ROWS

    # RandomState, rather than default_rng, is intentional: Isaac Gym calls
    # np.random.seed(seed), which selects this legacy MT19937 stream.
    rng = np.random.RandomState(seed)
    selected = {}
    wanted = set(STAGE_COLUMNS)
    # Terrain.curiculum() loops columns first, rows second.
    for col in range(NUM_COLS):
        choice = col / NUM_COLS + 0.001
        for terrain_row in range(NUM_ROWS):
            terrain_difficulty = terrain_row / NUM_ROWS
            raw, local_goals = _make_tile(
                rng, choice, terrain_difficulty,
                center_flat_goals=bool(center_flat_goals),
                step_x_range=step_x_range)
            if add_roughness:
                _add_random_roughness(raw, rng)
            if terrain_row == row and col in wanted:
                selected[col] = (raw.copy(), local_goals.copy())

    full_fields = [selected[col][0] for col in STAGE_COLUMNS]
    local_goal_sets = [selected[col][1] for col in STAGE_COLUMNS]
    fields = [
        _crop_after_final_goal(raw, goals)
        for raw, goals in zip(full_fields, local_goal_sets)
    ]
    _remove_internal_x_pads(fields)
    # Cropping discards the source tile's far-edge pad.  Restore it only at the
    # end of the complete campaign; all intermediate stage ends stay open.
    pad = max(1, int(_PAD_WIDTH / HORIZONTAL_SCALE))
    fields[-1][-pad:, :] = full_fields[-1][-pad:, :]

    # The down-stair generator descends from zero.  Lift the whole tile by the
    # nominal summit height so it starts where the up stairs end and returns to
    # zero (apart from the independently sampled centimetre-scale roughness).
    step_raw = int(
        (0.05 + difficulty * (0.20 - 0.05)) / VERTICAL_SCALE)
    summit_raw = 18 * step_raw
    fields[1][:] = (
        fields[1].astype(np.int32) + summit_raw).astype(np.int16)

    stages = []
    origin_x = 0.0
    for index, (col, kind, label, raw, local_goals) in enumerate(zip(
            STAGE_COLUMNS,
            STAGE_KINDS,
            STAGE_LABELS,
            fields,
            local_goal_sets)):
        goals = _goals_world(local_goals, raw, origin_x)
        stages.append(Stage(
            index=index,
            row=row,
            col=col,
            kind=kind,
            label=label,
            difficulty=difficulty,
            origin_x=origin_x,
            height_field_raw=raw,
            goals=goals,
            height_offset=(summit_raw * VERTICAL_SCALE
                           if kind == KIND_DOWN_STAIRS else 0.0),
        ))
        origin_x += raw.shape[0] * HORIZONTAL_SCALE

    combined = np.concatenate(fields, axis=0)
    return Campaign(
        stages=tuple(stages),
        height_field_raw=combined,
        horizontal_scale=HORIZONTAL_SCALE,
        vertical_scale=VERTICAL_SCALE,
        arena_length=ARENA_LENGTH,
        arena_width=ARENA_WIDTH,
        seed=seed,
        row=row,
        add_roughness=bool(add_roughness),
        center_flat_goals=bool(center_flat_goals),
        step_x_range=step_x_range,
    )


def _format_float(value: float) -> str:
    return "{:.12g}".format(float(value))


def _side_padding_nodes(campaign: Campaign, side_margin: float) -> int:
    side_margin = float(side_margin)
    if not np.isfinite(side_margin) or side_margin < 0.0:
        raise ValueError("MuJoCo campaign side margin must be non-negative")
    nodes = int(round(side_margin / campaign.horizontal_scale))
    if not np.isclose(
            nodes * campaign.horizontal_scale, side_margin,
            atol=1e-9, rtol=0.0):
        raise ValueError(
            "MuJoCo campaign side margin must align to horizontal_scale")
    return nodes


def _padded_hfield_heights(
        campaign: Campaign,
        side_margin: float = 0.0,
        remove_lateral_walls: bool = False,
        widen_stair_shoulders: bool = False) -> np.ndarray:
    """Move MuJoCo's unavoidable hfield side skirts away from the course.

    Edge replication adds solid shoulders and moves the distant hfield skirts
    out of the camera volume.  Isaac's parkour generator also puts a narrow
    0.5 m-high pad along both lateral edges.  For browser visualisation that
    pad looks like a continuous wall, especially after the downhill stage has
    accumulated a height offset.  ``remove_lateral_walls`` replaces only those
    outer pad samples with their immediately-adjacent terrain values.

    The original stair tiles deliberately have a narrow elevated lane with
    low terrain on either side.  A true side camera therefore sees a tall
    vertical slab near the summit even after the outer pad is removed.
    ``widen_stair_shoulders`` extends each stair tread sideways while retaining
    the central 1.0 m corridor sample-for-sample.  That corridor contains the
    policy's +/-0.4 m height scan and the robot's feet.  The replicated
    shoulder and distant catch plane continue to catch lateral falls.
    """
    nodes = _side_padding_nodes(campaign, side_margin)
    heights = campaign.heights
    if remove_lateral_walls or widen_stair_shoulders:
        heights = heights.copy()
    if remove_lateral_walls:
        wall_nodes = max(
            1, int(round(_PAD_WIDTH / campaign.horizontal_scale)))
        if heights.shape[1] <= 2 * wall_nodes:
            raise ValueError(
                "campaign is too narrow to remove lateral parkour walls")
        heights = heights.copy()
        heights[:, :wall_nodes] = heights[
            :, wall_nodes:wall_nodes + 1]
        heights[:, -wall_nodes:] = heights[
            :, -wall_nodes - 1:-wall_nodes]
    if widen_stair_shoulders:
        mid_y = heights.shape[1] // 2
        core_half_nodes = int(round(0.5 / campaign.horizontal_scale))
        core_start = mid_y - core_half_nodes
        core_stop = mid_y + core_half_nodes + 1
        if core_start < 0 or core_stop > heights.shape[1]:
            raise ValueError(
                "campaign is too narrow to retain the stair core corridor")
        for stage in campaign.stages:
            if stage.kind not in (KIND_UP_STAIRS, KIND_DOWN_STAIRS):
                continue
            start = int(round(stage.origin_x / campaign.horizontal_scale))
            stop = start + stage.height_field_raw.shape[0]
            original_core = heights[start:stop, core_start:core_stop].copy()
            center_profile = heights[start:stop, mid_y:mid_y + 1]
            heights[start:stop, :] = center_profile
            heights[start:stop, core_start:core_stop] = original_core
    if nodes == 0:
        return heights
    return np.pad(heights, ((0, 0), (nodes, nodes)), mode="edge")


def _hfield_placement(
        campaign: Campaign,
        side_margin: float = 0.0) -> Tuple[float, float, float, float]:
    """Return exact MuJoCo node radii and XY centre.

    A grid with ``n`` samples has ``n - 1`` intervals.  Using ``n * scale`` as
    the hfield diameter silently stretches every scan/collision coordinate.
    Isaac's y index zero is at ``-campaign.width / 2`` rather than at zero, so
    the 40-node grid is centred at -0.05 m.
    """

    side_nodes = _side_padding_nodes(campaign, side_margin)
    nx, original_ny = campaign.height_field_raw.shape
    ny = original_ny + 2 * side_nodes
    x_radius = (nx - 1) * campaign.horizontal_scale / 2.0
    y_radius = (ny - 1) * campaign.horizontal_scale / 2.0
    center_x = x_radius
    first_y = (
        -campaign.width / 2.0
        - side_nodes * campaign.horizontal_scale)
    center_y = first_y + y_radius
    return x_radius, y_radius, center_x, center_y


def build_mujoco_model(
        xml_path: str,
        campaign: Campaign,
        side_margin: float = 0.0,
        remove_lateral_walls: bool = False,
        widen_stair_shoulders: bool = False):
    """Load an N2 XML and replace its box stairs with the campaign hfield.

    MuJoCo is imported lazily so terrain generation and its unit tests work on
    Isaac-only machines.  The original top-level plane is retained exactly
    once, but moved one metre below the lowest height-field surface.  It acts
    only as a distant catch plane.  The new hfield data is populated after
    ``MjModel.from_xml_string`` creates its empty storage.
    """

    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "build_mujoco_model requires the optional 'mujoco' package") from exc

    xml_file = Path(xml_path).expanduser().resolve()
    root = ET.parse(str(xml_file)).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    meshdir = compiler.get("meshdir")
    if meshdir:
        mesh_path = Path(meshdir)
        if not mesh_path.is_absolute():
            mesh_path = (xml_file.parent / mesh_path).resolve()
        compiler.set("meshdir", str(mesh_path))

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        insert_at = 1 if root.find("compiler") is not None else 0
        root.insert(insert_at, asset)
    for old in list(asset.findall("hfield")):
        if old.get("name") == "parkour_campaign":
            asset.remove(old)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo XML has no <worldbody>")
    direct_geoms = list(worldbody.findall("geom"))
    planes = [geom for geom in direct_geoms
              if geom.get("type", "sphere") == "plane"]
    if len(planes) != 1:
        raise ValueError(
            "expected exactly one top-level plane, found {}".format(
                len(planes)))
    plane = planes[0]
    for geom in direct_geoms:
        if geom.get("type", "sphere") == "box":
            worldbody.remove(geom)
        elif (geom.get("type", "sphere") == "hfield"
              and geom.get("hfield") == "parkour_campaign"):
            worldbody.remove(geom)

    heights = _padded_hfield_heights(
        campaign,
        side_margin,
        remove_lateral_walls=remove_lateral_walls,
        widen_stair_shoulders=widen_stair_shoulders,
    )
    minimum = float(np.min(heights))
    maximum = float(np.max(heights))
    height_range = max(maximum - minimum, campaign.vertical_scale)
    plane_pos = np.fromstring(plane.get("pos", "0 0 0"), sep=" ")
    if plane_pos.size != 3:
        plane_pos = np.zeros(3, dtype=np.float64)
    plane_pos[2] = minimum - 1.0
    plane.set("pos", " ".join(_format_float(v) for v in plane_pos))

    # MuJoCo height fields use rows along y and columns along x, hence the
    # transpose relative to Isaac's [x, y] array.
    nrow = heights.shape[1]
    ncol = heights.shape[0]
    x_radius, y_radius, center_x, center_y = _hfield_placement(
        campaign, side_margin)
    hfield = ET.SubElement(asset, "hfield", {
        "name": "parkour_campaign",
        "nrow": str(nrow),
        "ncol": str(ncol),
        "size": "{} {} {} {}".format(
            _format_float(x_radius),
            _format_float(y_radius),
            _format_float(height_range),
            _format_float(max(0.1, -minimum + 0.1)),
        ),
    })
    del hfield  # ElementTree retains it; silence linters about the local name.

    geom_attributes = {
        "name": "parkour_campaign_geom",
        "type": "hfield",
        "hfield": "parkour_campaign",
        "pos": "{} {} {}".format(
            _format_float(center_x),
            _format_float(center_y),
            _format_float(minimum),
        ),
    }
    for attribute in ("friction", "material", "solref", "solimp",
                      "condim", "contype", "conaffinity"):
        value = plane.get(attribute)
        if value is not None:
            geom_attributes[attribute] = value
    ET.SubElement(worldbody, "geom", geom_attributes)

    xml_text = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_text)
    hfield_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_HFIELD, "parkour_campaign")
    if hfield_id < 0:
        raise RuntimeError("MuJoCo did not create parkour_campaign hfield")
    address = int(model.hfield_adr[hfield_id])
    count = int(model.hfield_nrow[hfield_id] * model.hfield_ncol[hfield_id])
    normalised = ((heights - minimum) / height_range).T.ravel()
    if normalised.size != count:
        raise RuntimeError(
            "hfield storage mismatch: model {}, terrain {}".format(
                count, normalised.size))
    model.hfield_data[address:address + count] = normalised
    return model


__all__ = [
    "ARENA_LENGTH",
    "ARENA_WIDTH",
    "Campaign",
    "HORIZONTAL_SCALE",
    "KIND_DOWN_STAIRS",
    "KIND_FLAT",
    "KIND_HURDLE",
    "KIND_STEPPING_STONES",
    "KIND_UP_STAIRS",
    "NUM_GOALS",
    "STAGE_COLUMNS",
    "STAGE_KINDS",
    "Stage",
    "VERTICAL_SCALE",
    "build_campaign",
    "build_mujoco_model",
    "sample_heights",
]
