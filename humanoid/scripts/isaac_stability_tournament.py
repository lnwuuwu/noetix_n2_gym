"""Choose a smoother Isaac stair policy without sacrificing climbing.

The tournament compares complete 0--10 cm evaluation tables generated from
the same deterministic seed. Completion/fall checks are expressed in whole
episode resolution, while selection is driven by action motion, left/right
swing balance, foot lanes, support-phase shake, and lateral drift.  The
default targeted mode also makes the three video-visible defects hard gates
at 10 cm: lateral translation, unequal strides, and right-support shake.
"""

import argparse
import csv
import json
import math
import os


LEVELS = tuple(range(5))
# Half of training environments stay on 10 cm stairs.  Use the same emphasis
# when aggregating evaluation metrics.
LEVEL_WEIGHTS = {
    0: 0.125,
    1: 0.125,
    2: 0.125,
    3: 0.125,
    4: 0.500,
}


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return float(default)
    return float(value)


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return int(default)
    return int(value)


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).lower() in {"1", "true", "yes", "on"}


def compare_config():
    """Return safety/style gates for deterministic checkpoint selection.

    ``targeted`` is the default for long polishing and requires measurable
    improvement in at least two of lateral translation, stride balance, and
    right-support stability at 10 cm. ``strict`` retains the older global
    all-or-nothing style guard. ``balanced`` remains available for diagnostic
    searches that intentionally permit a safe fallback.
    """
    mode = os.getenv(
        "N2_ISAAC_STABILITY_SELECTION_MODE", "targeted"
    ).strip().lower()
    presets = {
        "targeted": {
            "high_tolerance_min": 0.020,
            "level_tolerance_min": 0.080,
            "completion_regression_tol": 0.020,
            "fall_tolerance": 0.020,
            "path_tolerance": 0.020,
            "command_error_tolerance": 0.020,
            "max_lateral_tol": 0.004,
            "yaw_tol": 0.020,
            "action_tol_ratio": 1.02,
            "action_abs": 0.005,
            "stride_imbalance_tol": 0.005,
            "signed_lateral_tol": 0.004,
            "foot_inward_tol": 0.003,
            "foot_lane_center_tol": 0.003,
            "left_swing_action_ratio": 1.01,
            "left_swing_action_abs": 0.004,
            "left_swing_body_ratio": 1.02,
            "left_swing_body_abs": 0.004,
            "actor_symmetry_ratio": 1.05,
            "actor_symmetry_abs": 0.002,
            "material_min_required": 3,
            "hard_min_level": 2,
            "minimum_style_gain_abs": 0.005,
            "minimum_style_gain_rel": 0.002,
            "allow_safe_fallback": False,
            "fallback_min_style_gain": 0.0,
            "fallback_min_score": 0.0,
            "fallback_material_min": 2,
            "enforce_targeted_gait": True,
            "target_min_groups": 2,
            "target_max_lateral_tol": 0.002,
            "target_signed_lateral_tol": 0.002,
            "target_stride_tol": 0.004,
            "target_foot_lane_tol": 0.002,
            "target_left_action_ratio": 1.01,
            "target_left_action_abs": 0.003,
            "target_left_body_ratio": 1.02,
            "target_left_body_abs": 0.003,
            "target_lateral_improvement": 0.002,
            "target_stride_improvement": 0.005,
            "target_shake_improvement": 0.008,
        },
        "strict": {
            "high_tolerance_min": 0.025,
            "level_tolerance_min": 0.080,
            "completion_regression_tol": 0.025,
            "fall_tolerance": 0.025,
            "path_tolerance": 0.025,
            "command_error_tolerance": 0.020,
            "max_lateral_tol": 0.008,
            "yaw_tol": 0.020,
            "action_tol_ratio": 1.02,
            "action_abs": 0.005,
            "stride_imbalance_tol": 0.008,
            "signed_lateral_tol": 0.008,
            "foot_inward_tol": 0.004,
            "foot_lane_center_tol": 0.004,
            "left_swing_action_ratio": 1.03,
            "left_swing_action_abs": 0.005,
            "left_swing_body_ratio": 1.05,
            "left_swing_body_abs": 0.005,
            "actor_symmetry_ratio": 1.05,
            "actor_symmetry_abs": 0.002,
            "material_min_required": 2,
            "hard_min_level": 0,
            "minimum_style_gain_abs": 0.010,
            "minimum_style_gain_rel": 0.010,
            "allow_safe_fallback": False,
            "fallback_min_style_gain": 0.0,
            "fallback_min_score": 0.0,
            "fallback_material_min": 1,
        },
        "balanced": {
            "high_tolerance_min": 0.030,
            "level_tolerance_min": 0.100,
            "completion_regression_tol": 0.030,
            "fall_tolerance": 0.025,
            "path_tolerance": 0.025,
            "command_error_tolerance": 0.030,
            "max_lateral_tol": 0.012,
            "yaw_tol": 0.030,
            "action_tol_ratio": 1.04,
            "action_abs": 0.010,
            "stride_imbalance_tol": 0.012,
            "signed_lateral_tol": 0.012,
            "foot_inward_tol": 0.006,
            "foot_lane_center_tol": 0.006,
            "left_swing_action_ratio": 1.04,
            "left_swing_action_abs": 0.008,
            "left_swing_body_ratio": 1.05,
            "left_swing_body_abs": 0.008,
            "actor_symmetry_ratio": 1.08,
            "actor_symmetry_abs": 0.003,
            "material_min_required": 1,
            "hard_min_level": 2,
            "minimum_style_gain_abs": 0.003,
            "minimum_style_gain_rel": 0.001,
            "allow_safe_fallback": True,
            "fallback_min_style_gain": 0.001,
            "fallback_min_score": 0.001,
            "fallback_material_min": 1,
        },
    }
    if mode not in presets:
        raise ValueError(
            "N2_ISAAC_STABILITY_SELECTION_MODE must be targeted, strict, "
            "or balanced"
        )
    defaults = presets[mode]
    return {
        "mode": mode,
        "high_tolerance_min": _env_float(
            "N2_ISAAC_STABILITY_HIGH_TOL",
            defaults["high_tolerance_min"],
        ),
        "level_tolerance_min": _env_float(
            "N2_ISAAC_STABILITY_LEVEL_TOL",
            defaults["level_tolerance_min"],
        ),
        "completion_regression_tol": _env_float(
            "N2_ISAAC_STABILITY_COMPLETION_TOL",
            defaults["completion_regression_tol"],
        ),
        "fall_tolerance": _env_float(
            "N2_ISAAC_STABILITY_FALL_TOL", defaults["fall_tolerance"]
        ),
        "path_tolerance": _env_float(
            "N2_ISAAC_STABILITY_PATH_TOL", defaults["path_tolerance"]
        ),
        "command_error_tolerance": _env_float(
            "N2_ISAAC_STABILITY_CMDERR_TOL",
            defaults["command_error_tolerance"],
        ),
        "max_lateral_tol": _env_float(
            "N2_ISAAC_STABILITY_MAX_LAT_TOL",
            defaults["max_lateral_tol"],
        ),
        "yaw_tol": _env_float(
            "N2_ISAAC_STABILITY_YAW_TOL", defaults["yaw_tol"]
        ),
        "action_tol_ratio": _env_float(
            "N2_ISAAC_STABILITY_ACTION_RATIO",
            defaults["action_tol_ratio"],
        ),
        "action_abs": _env_float(
            "N2_ISAAC_STABILITY_ACTION_ABS", defaults["action_abs"]
        ),
        "stride_imbalance_tol": _env_float(
            "N2_ISAAC_STABILITY_STRIDE_IMB_TOL",
            defaults["stride_imbalance_tol"],
        ),
        "signed_lateral_tol": _env_float(
            "N2_ISAAC_STABILITY_SIGNED_LAT_TOL",
            defaults["signed_lateral_tol"],
        ),
        "foot_inward_tol": _env_float(
            "N2_ISAAC_STABILITY_FOOT_INWARD_TOL",
            defaults["foot_inward_tol"],
        ),
        "foot_lane_center_tol": _env_float(
            "N2_ISAAC_STABILITY_LANE_CENTER_TOL",
            defaults["foot_lane_center_tol"],
        ),
        "left_swing_action_ratio": _env_float(
            "N2_ISAAC_STABILITY_LEFT_SWING_ACTION_RATIO",
            defaults["left_swing_action_ratio"],
        ),
        "left_swing_action_abs": _env_float(
            "N2_ISAAC_STABILITY_LEFT_SWING_ACTION_ABS",
            defaults["left_swing_action_abs"],
        ),
        "left_swing_body_ratio": _env_float(
            "N2_ISAAC_STABILITY_LEFT_SWING_BODY_RATIO",
            defaults["left_swing_body_ratio"],
        ),
        "left_swing_body_abs": _env_float(
            "N2_ISAAC_STABILITY_LEFT_SWING_BODY_ABS",
            defaults["left_swing_body_abs"],
        ),
        "actor_symmetry_ratio": _env_float(
            "N2_ISAAC_STABILITY_ACTOR_SYM_RATIO",
            defaults["actor_symmetry_ratio"],
        ),
        "actor_symmetry_abs": _env_float(
            "N2_ISAAC_STABILITY_ACTOR_SYM_ABS",
            defaults["actor_symmetry_abs"],
        ),
        "material_min_required": _env_int(
            "N2_ISAAC_STABILITY_MATERIAL_MIN",
            defaults["material_min_required"],
        ),
        "hard_min_level": _env_int(
            "N2_ISAAC_STABILITY_HARD_MIN_LEVEL",
            defaults["hard_min_level"],
        ),
        "minimum_style_gain_abs": _env_float(
            "N2_ISAAC_STABILITY_STYLE_GAIN_MIN",
            defaults["minimum_style_gain_abs"],
        ),
        "minimum_style_gain_rel": _env_float(
            "N2_ISAAC_STABILITY_STYLE_GAIN_REL",
            defaults["minimum_style_gain_rel"],
        ),
        "enforce_targeted_gait": _env_bool(
            "N2_ISAAC_STABILITY_ENFORCE_TARGETED",
            defaults.get("enforce_targeted_gait", False),
        ),
        "target_min_groups": _env_int(
            "N2_ISAAC_STABILITY_TARGET_MIN_GROUPS",
            defaults.get("target_min_groups", 0),
        ),
        "target_max_lateral_tol": _env_float(
            "N2_ISAAC_STABILITY_TARGET_MAX_LAT_TOL",
            defaults.get("target_max_lateral_tol", 0.0),
        ),
        "target_signed_lateral_tol": _env_float(
            "N2_ISAAC_STABILITY_TARGET_SIGNED_LAT_TOL",
            defaults.get("target_signed_lateral_tol", 0.0),
        ),
        "target_stride_tol": _env_float(
            "N2_ISAAC_STABILITY_TARGET_STRIDE_TOL",
            defaults.get("target_stride_tol", 0.0),
        ),
        "target_foot_lane_tol": _env_float(
            "N2_ISAAC_STABILITY_TARGET_FOOT_LANE_TOL",
            defaults.get("target_foot_lane_tol", 0.0),
        ),
        "target_left_action_ratio": _env_float(
            "N2_ISAAC_STABILITY_TARGET_LEFT_ACTION_RATIO",
            defaults.get("target_left_action_ratio", 1.0),
        ),
        "target_left_action_abs": _env_float(
            "N2_ISAAC_STABILITY_TARGET_LEFT_ACTION_ABS",
            defaults.get("target_left_action_abs", 0.0),
        ),
        "target_left_body_ratio": _env_float(
            "N2_ISAAC_STABILITY_TARGET_LEFT_BODY_RATIO",
            defaults.get("target_left_body_ratio", 1.0),
        ),
        "target_left_body_abs": _env_float(
            "N2_ISAAC_STABILITY_TARGET_LEFT_BODY_ABS",
            defaults.get("target_left_body_abs", 0.0),
        ),
        "target_lateral_improvement": _env_float(
            "N2_ISAAC_STABILITY_TARGET_LATERAL_GAIN",
            defaults.get("target_lateral_improvement", 0.0),
        ),
        "target_stride_improvement": _env_float(
            "N2_ISAAC_STABILITY_TARGET_STRIDE_GAIN",
            defaults.get("target_stride_improvement", 0.0),
        ),
        "target_shake_improvement": _env_float(
            "N2_ISAAC_STABILITY_TARGET_SHAKE_GAIN",
            defaults.get("target_shake_improvement", 0.0),
        ),
        "weight_completion": _env_float(
            "N2_ISAAC_STABILITY_WEIGHT_COMPLETION", 4.0
        ),
        "weight_fall": _env_float("N2_ISAAC_STABILITY_WEIGHT_FALL", 3.0),
        "weight_path": _env_float("N2_ISAAC_STABILITY_WEIGHT_PATH", 2.0),
        "reject_on_negative_performance": _env_bool(
            "N2_ISAAC_STABILITY_REJECT_NEG_PERF", False
        ),
        "allow_safe_fallback": _env_bool(
            "N2_ISAAC_STABILITY_ALLOW_SAFE_FALLBACK",
            defaults["allow_safe_fallback"],
        ),
        "fallback_min_style_gain": _env_float(
            "N2_ISAAC_STABILITY_FALLBACK_STYLE_GAIN",
            defaults["fallback_min_style_gain"],
        ),
        "fallback_min_score": _env_float(
            "N2_ISAAC_STABILITY_FALLBACK_SCORE",
            defaults["fallback_min_score"],
        ),
        "fallback_material_min": _env_int(
            "N2_ISAAC_STABILITY_FALLBACK_MATERIAL_MIN",
            defaults["fallback_material_min"],
        ),
    }


def load_evaluation(path):
    with open(path, newline="") as csv_file:
        source_rows = list(csv.DictReader(csv_file))
    rows = {}
    for source in source_rows:
        level = int(float(source["terrain_level"]))
        if level in rows:
            raise ValueError(
                "{} contains duplicate terrain level {}".format(path, level)
            )
        converted = {}
        for key, value in source.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                converted[key] = value
                continue
            if not math.isfinite(numeric):
                raise ValueError("{} has non-finite {}".format(path, key))
            converted[key] = numeric
        rows[level] = converted
    missing = [level for level in LEVELS if level not in rows]
    if missing:
        raise ValueError(
            "{} is missing terrain levels {}".format(
                path, ",".join(str(level) for level in missing)
            )
        )
    return rows


def weighted_mean(rows, key):
    return sum(
        LEVEL_WEIGHTS[level] * float(rows[level][key])
        for level in LEVELS
    )


def action_motion(row):
    return (
        float(row["mean_action_rate_rms"])
        + 0.5 * float(row["mean_action_accel_rms"])
    )


def optional_float(row, key, fallback):
    value = row.get(key, fallback)
    return float(value)


def phase_action_motion(row, swing_side):
    """Return action motion during one swing/support half-cycle."""
    return (
        optional_float(
            row,
            "mean_{}_swing_action_rate_rms".format(swing_side),
            row["mean_action_rate_rms"],
        )
        + 0.5
        * optional_float(
            row,
            "mean_{}_swing_action_accel_rms".format(swing_side),
            row["mean_action_accel_rms"],
        )
    )


def phase_body_motion(row, swing_side):
    """Return roll/lateral body motion during one support half-cycle."""
    return (
        optional_float(
            row,
            "mean_{}_swing_roll_rate_rms".format(swing_side),
            0.0,
        )
        + optional_float(
            row,
            "mean_{}_swing_lateral_velocity_rms".format(swing_side),
            0.0,
        )
    )


def stride_imbalance(row):
    return abs(
        float(row["mean_left_swing_length_m"])
        - float(row["mean_right_swing_length_m"])
    )


def foot_lane_center_error(row):
    """Return common-mode left/right foot-lane translation."""
    return abs(
        0.5
        * (
            optional_float(
                row, "mean_left_foot_lateral_position_m", 0.0
            )
            + optional_float(
                row, "mean_right_foot_lateral_position_m", 0.0
            )
        )
    )


def targeted_gait_metrics(row):
    """Metrics for the three defects identified in the recorded rollout."""
    return {
        "max_lateral": float(row["mean_max_lateral_deviation_m"]),
        "signed_lateral": abs(
            float(row["mean_final_lateral_position_m"])
        ),
        "foot_lane_center": foot_lane_center_error(row),
        "stride_imbalance": stride_imbalance(row),
        # Left swing is the observed right-foot-support shake phase.
        "left_swing_action_motion": phase_action_motion(row, "left"),
        "left_swing_body_motion": phase_body_motion(row, "left"),
    }


def aggregate(rows):
    action = sum(
        LEVEL_WEIGHTS[level] * action_motion(rows[level])
        for level in LEVELS
    )
    stride = sum(
        LEVEL_WEIGHTS[level] * stride_imbalance(rows[level])
        for level in LEVELS
    )
    signed_lateral = sum(
        LEVEL_WEIGHTS[level]
        * abs(float(rows[level]["mean_final_lateral_position_m"]))
        for level in LEVELS
    )
    max_lateral = weighted_mean(rows, "mean_max_lateral_deviation_m")
    yaw = weighted_mean(rows, "mean_max_yaw_deviation_rad")
    double_flight = weighted_mean(rows, "mean_double_flight_fraction")
    actor_symmetry_error = sum(
        LEVEL_WEIGHTS[level]
        * optional_float(
            rows[level], "mean_actor_symmetry_error_rms", 0.0
        )
        for level in LEVELS
    )
    left_inward = sum(
        LEVEL_WEIGHTS[level]
        * optional_float(
            rows[level], "mean_left_foot_inward_error_m", 0.0
        )
        for level in LEVELS
    )
    right_inward = sum(
        LEVEL_WEIGHTS[level]
        * optional_float(
            rows[level], "mean_right_foot_inward_error_m", 0.0
        )
        for level in LEVELS
    )
    foot_inward = left_inward + right_inward
    # A centered gait has y_left ~= +offset and y_right ~= -offset.  The
    # previous gate watched only right-foot crossover, but the measured
    # failure is a common rightward foot-lane shift: the left foot approaches
    # the centerline while the right foot stays far outside it.
    foot_lane_center = sum(
        LEVEL_WEIGHTS[level]
        * foot_lane_center_error(rows[level])
        for level in LEVELS
    )
    foot_lane_half_width_error = sum(
        LEVEL_WEIGHTS[level]
        * abs(
            0.5
            * (
                optional_float(
                    rows[level],
                    "mean_left_foot_lateral_position_m",
                    0.09,
                )
                - optional_float(
                    rows[level],
                    "mean_right_foot_lateral_position_m",
                    -0.09,
                )
            )
            - 0.09
        )
        for level in LEVELS
    )
    right_swing_motion = sum(
        LEVEL_WEIGHTS[level]
        * phase_action_motion(rows[level], "right")
        for level in LEVELS
    )
    # Left swing means the right foot is the sole scheduled support.  This is
    # the phase in which the physical robot visibly shakes.
    left_swing_motion = sum(
        LEVEL_WEIGHTS[level]
        * phase_action_motion(rows[level], "left")
        for level in LEVELS
    )
    phase_motion_imbalance = abs(
        left_swing_motion - right_swing_motion
    )
    right_swing_body_motion = sum(
        LEVEL_WEIGHTS[level]
        * phase_body_motion(rows[level], "right")
        for level in LEVELS
    )
    left_swing_body_motion = sum(
        LEVEL_WEIGHTS[level]
        * phase_body_motion(rows[level], "left")
        for level in LEVELS
    )
    phase_body_imbalance = abs(
        left_swing_body_motion - right_swing_body_motion
    )
    style_cost = (
        action
        + 8.0 * stride
        + 4.0 * signed_lateral
        + 2.0 * max_lateral
        + 0.25 * yaw
        + 0.5 * double_flight
        + 8.0 * foot_inward
        + 8.0 * foot_lane_center
        + 2.0 * foot_lane_half_width_error
        + left_swing_motion
        + 1.5 * phase_motion_imbalance
        + 2.0 * left_swing_body_motion
        + phase_body_imbalance
        + 0.25 * actor_symmetry_error
    )
    return {
        "completion": weighted_mean(rows, "completion_rate"),
        "fall": weighted_mean(rows, "fall_rate"),
        "path_failure": weighted_mean(rows, "path_failure_rate"),
        "command_error": weighted_mean(rows, "mean_command_error_m_s"),
        "action_motion": action,
        "stride_imbalance": stride,
        "signed_lateral": signed_lateral,
        "max_lateral": max_lateral,
        "yaw": yaw,
        "double_flight": double_flight,
        "actor_symmetry_error": actor_symmetry_error,
        "left_foot_inward": left_inward,
        "right_foot_inward": right_inward,
        "foot_inward": foot_inward,
        "foot_lane_center": foot_lane_center,
        "foot_lane_half_width_error": foot_lane_half_width_error,
        "right_swing_action_motion": right_swing_motion,
        "left_swing_action_motion": left_swing_motion,
        "phase_action_imbalance": phase_motion_imbalance,
        "right_swing_body_motion": right_swing_body_motion,
        "left_swing_body_motion": left_swing_body_motion,
        "phase_body_imbalance": phase_body_imbalance,
        "style_cost": style_cost,
    }


def episode_tolerance(baseline_row, candidate_row, floor, episodes):
    observed = min(
        float(baseline_row.get("episodes", episodes)),
        float(candidate_row.get("episodes", episodes)),
        float(episodes),
    )
    return max(float(floor), 2.0 / max(observed, 1.0))


def compare(baseline_rows, candidate_rows, episodes=None):
    cfg = compare_config()
    baseline = aggregate(baseline_rows)
    candidate = aggregate(candidate_rows)
    if episodes is None:
        episodes = min(
            int(float(baseline_rows[level].get("episodes", 1)))
            for level in LEVELS
        )
    hard_reasons = []
    soft_reasons = []

    high_baseline = baseline_rows[4]
    high_candidate = candidate_rows[4]
    high_tolerance = max(
        episode_tolerance(
            high_baseline, high_candidate, 0.025, episodes
        ),
        cfg["high_tolerance_min"],
    )
    if high_candidate["completion_rate"] < (
        high_baseline["completion_rate"] - high_tolerance
    ):
        hard_reasons.append(
            "10 cm completion regressed {:.3%} -> {:.3%} "
            "(tolerance {:.3%})".format(
                high_baseline["completion_rate"],
                high_candidate["completion_rate"],
                high_tolerance,
            )
        )
    if high_candidate["fall_rate"] > (
        high_baseline["fall_rate"] + high_tolerance
    ):
        hard_reasons.append(
            "10 cm fall increased {:.3%} -> {:.3%} "
            "(tolerance {:.3%})".format(
                high_baseline["fall_rate"],
                high_candidate["fall_rate"],
                high_tolerance,
            )
        )
    if high_candidate["path_failure_rate"] > (
        high_baseline["path_failure_rate"] + high_tolerance
    ):
        hard_reasons.append(
            "10 cm path failure increased {:.3%} -> {:.3%}".format(
                high_baseline["path_failure_rate"],
                high_candidate["path_failure_rate"],
            )
        )

    high_target_baseline = targeted_gait_metrics(high_baseline)
    high_target_candidate = targeted_gait_metrics(high_candidate)
    target_deltas = {
        name: high_target_baseline[name] - high_target_candidate[name]
        for name in high_target_baseline
    }
    target_group_deltas = {
        "lateral": max(
            target_deltas["max_lateral"],
            target_deltas["signed_lateral"],
            target_deltas["foot_lane_center"],
        ),
        "stride": target_deltas["stride_imbalance"],
        "right_support_shake": max(
            target_deltas["left_swing_action_motion"],
            target_deltas["left_swing_body_motion"],
        ),
    }
    target_group_improvements = {
        "lateral": (
            target_group_deltas["lateral"]
            >= cfg["target_lateral_improvement"]
        ),
        "stride": (
            target_group_deltas["stride"]
            >= cfg["target_stride_improvement"]
        ),
        "right_support_shake": (
            target_group_deltas["right_support_shake"]
            >= cfg["target_shake_improvement"]
        ),
    }
    target_groups_improved = sum(target_group_improvements.values())

    if cfg["enforce_targeted_gait"]:
        if high_target_candidate["max_lateral"] > (
            high_target_baseline["max_lateral"]
            + cfg["target_max_lateral_tol"]
        ):
            hard_reasons.append(
                "10 cm lateral deviation increased {:.4f} -> {:.4f}".format(
                    high_target_baseline["max_lateral"],
                    high_target_candidate["max_lateral"],
                )
            )
        if high_target_candidate["signed_lateral"] > (
            high_target_baseline["signed_lateral"]
            + cfg["target_signed_lateral_tol"]
        ):
            hard_reasons.append(
                "10 cm signed lateral error increased {:.4f} -> {:.4f}".format(
                    high_target_baseline["signed_lateral"],
                    high_target_candidate["signed_lateral"],
                )
            )
        if high_target_candidate["foot_lane_center"] > (
            high_target_baseline["foot_lane_center"]
            + cfg["target_foot_lane_tol"]
        ):
            hard_reasons.append(
                "10 cm foot-lane center error increased {:.4f} -> {:.4f}".format(
                    high_target_baseline["foot_lane_center"],
                    high_target_candidate["foot_lane_center"],
                )
            )
        if high_target_candidate["stride_imbalance"] > (
            high_target_baseline["stride_imbalance"]
            + cfg["target_stride_tol"]
        ):
            hard_reasons.append(
                "10 cm stride imbalance increased {:.4f} -> {:.4f}".format(
                    high_target_baseline["stride_imbalance"],
                    high_target_candidate["stride_imbalance"],
                )
            )
        if high_target_candidate["left_swing_action_motion"] > (
            high_target_baseline["left_swing_action_motion"]
            * cfg["target_left_action_ratio"]
            + cfg["target_left_action_abs"]
        ):
            hard_reasons.append(
                "10 cm right-support/left-swing action motion increased "
                "{:.4f} -> {:.4f}".format(
                    high_target_baseline["left_swing_action_motion"],
                    high_target_candidate["left_swing_action_motion"],
                )
            )
        if high_target_candidate["left_swing_body_motion"] > (
            high_target_baseline["left_swing_body_motion"]
            * cfg["target_left_body_ratio"]
            + cfg["target_left_body_abs"]
        ):
            hard_reasons.append(
                "10 cm right-support/left-swing body motion increased "
                "{:.4f} -> {:.4f}".format(
                    high_target_baseline["left_swing_body_motion"],
                    high_target_candidate["left_swing_body_motion"],
                )
            )
        if target_groups_improved < cfg["target_min_groups"]:
            hard_reasons.append(
                "10 cm improved only {} of 3 target defect groups; "
                "{} required".format(
                    target_groups_improved, cfg["target_min_groups"]
                )
            )

    for level in LEVELS:
        level_baseline = baseline_rows[level]
        level_candidate = candidate_rows[level]
        level_reasons = (
            hard_reasons
            if level >= cfg["hard_min_level"]
            else soft_reasons
        )
        level_tolerance = max(
            episode_tolerance(
                level_baseline, level_candidate, 0.08, episodes
            ),
            cfg["level_tolerance_min"],
        )
        if level_candidate["completion_rate"] < (
            level_baseline["completion_rate"] - level_tolerance
        ):
            level_reasons.append(
                "level {} completion regressed {:.3%} -> {:.3%}".format(
                    level,
                    level_baseline["completion_rate"],
                    level_candidate["completion_rate"],
                )
            )
        if level_candidate["fall_rate"] > (
            level_baseline["fall_rate"] + level_tolerance
        ):
            level_reasons.append(
                "level {} fall increased {:.3%} -> {:.3%}".format(
                    level,
                    level_baseline["fall_rate"],
                    level_candidate["fall_rate"],
                )
            )

    if candidate["completion"] < baseline["completion"] - cfg[
        "completion_regression_tol"
    ]:
        hard_reasons.append(
            "weighted completion regressed {:.3%} -> {:.3%}".format(
                baseline["completion"], candidate["completion"]
            )
        )
    if candidate["fall"] > baseline["fall"] + cfg["fall_tolerance"]:
        hard_reasons.append(
            "weighted fall increased {:.3%} -> {:.3%}".format(
                baseline["fall"], candidate["fall"]
            )
        )
    if candidate["path_failure"] > (
        baseline["path_failure"] + cfg["path_tolerance"]
    ):
        hard_reasons.append(
            "weighted path failure increased {:.3%} -> {:.3%}".format(
                baseline["path_failure"], candidate["path_failure"]
            )
        )
    if candidate["command_error"] > (
        baseline["command_error"] + cfg["command_error_tolerance"]
    ):
        soft_reasons.append(
            "weighted command error increased {:.4f} -> {:.4f}".format(
                baseline["command_error"], candidate["command_error"]
            )
        )
    if candidate["max_lateral"] > (
        baseline["max_lateral"] + cfg["max_lateral_tol"]
    ):
        soft_reasons.append(
            "weighted lateral deviation increased {:.4f} -> {:.4f}".format(
                baseline["max_lateral"], candidate["max_lateral"]
            )
        )
    if candidate["yaw"] > baseline["yaw"] + cfg["yaw_tol"]:
        soft_reasons.append(
            "weighted yaw deviation increased {:.4f} -> {:.4f}".format(
                baseline["yaw"], candidate["yaw"]
            )
        )
    if candidate["action_motion"] > (
        baseline["action_motion"] * cfg["action_tol_ratio"] + cfg["action_abs"]
    ):
        soft_reasons.append(
            "action motion increased {:.4f} -> {:.4f}".format(
                baseline["action_motion"], candidate["action_motion"]
            )
        )
    if candidate["stride_imbalance"] > (
        baseline["stride_imbalance"] + cfg["stride_imbalance_tol"]
    ):
        soft_reasons.append(
            "stride imbalance increased {:.4f} -> {:.4f}".format(
                baseline["stride_imbalance"],
                candidate["stride_imbalance"],
            )
        )
    if candidate["signed_lateral"] > (
        baseline["signed_lateral"] + cfg["signed_lateral_tol"]
    ):
        soft_reasons.append(
            "signed lateral error increased {:.4f} -> {:.4f}".format(
                baseline["signed_lateral"], candidate["signed_lateral"]
            )
        )
    if candidate["foot_inward"] > (
        baseline["foot_inward"] + cfg["foot_inward_tol"]
    ):
        soft_reasons.append(
            "combined foot inward error increased {:.4f} -> {:.4f}".format(
                baseline["foot_inward"],
                candidate["foot_inward"],
            )
        )
    if candidate["foot_lane_center"] > (
        baseline["foot_lane_center"] + cfg["foot_lane_center_tol"]
    ):
        soft_reasons.append(
            "foot-lane center error increased {:.4f} -> {:.4f}".format(
                baseline["foot_lane_center"],
                candidate["foot_lane_center"],
            )
        )
    if candidate["left_swing_action_motion"] > (
        baseline["left_swing_action_motion"] * cfg["left_swing_action_ratio"]
        + cfg["left_swing_action_abs"]
    ):
        soft_reasons.append(
            "right-support/left-swing action motion increased "
            "{:.4f} -> {:.4f}".format(
                baseline["left_swing_action_motion"],
                candidate["left_swing_action_motion"],
            )
        )
    if candidate["left_swing_body_motion"] > (
        baseline["left_swing_body_motion"] * cfg["left_swing_body_ratio"]
        + cfg["left_swing_body_abs"]
    ):
        soft_reasons.append(
            "right-support/left-swing body motion increased "
            "{:.4f} -> {:.4f}".format(
                baseline["left_swing_body_motion"],
                candidate["left_swing_body_motion"],
            )
        )
    if candidate["actor_symmetry_error"] > (
        baseline["actor_symmetry_error"] * cfg["actor_symmetry_ratio"]
        + cfg["actor_symmetry_abs"]
    ):
        soft_reasons.append(
            "Actor reflection error increased {:.4f} -> {:.4f}".format(
                baseline["actor_symmetry_error"],
                candidate["actor_symmetry_error"],
            )
        )

    improvements = {
        "action_motion": (
            baseline["action_motion"] - candidate["action_motion"]
        ),
        "stride_imbalance": (
            baseline["stride_imbalance"] - candidate["stride_imbalance"]
        ),
        "signed_lateral": (
            baseline["signed_lateral"] - candidate["signed_lateral"]
        ),
        "max_lateral": baseline["max_lateral"] - candidate["max_lateral"],
        "foot_inward": (
            baseline["foot_inward"] - candidate["foot_inward"]
        ),
        "foot_lane_center": (
            baseline["foot_lane_center"] - candidate["foot_lane_center"]
        ),
        "right_support_motion": (
            baseline["left_swing_action_motion"]
            - candidate["left_swing_action_motion"]
        ),
        "phase_action_imbalance": (
            baseline["phase_action_imbalance"]
            - candidate["phase_action_imbalance"]
        ),
        "right_support_body_motion": (
            baseline["left_swing_body_motion"]
            - candidate["left_swing_body_motion"]
        ),
        "phase_body_imbalance": (
            baseline["phase_body_imbalance"]
            - candidate["phase_body_imbalance"]
        ),
        "actor_symmetry_error": (
            baseline["actor_symmetry_error"]
            - candidate["actor_symmetry_error"]
        ),
    }
    material_improvements = sum(
        (
            improvements["action_motion"] >= 0.005,
            improvements["stride_imbalance"] >= 0.002,
            improvements["signed_lateral"] >= 0.002,
            improvements["max_lateral"] >= 0.002,
            improvements["foot_inward"] >= 0.001,
            improvements["foot_lane_center"] >= 0.001,
            improvements["right_support_motion"] >= 0.005,
            improvements["phase_action_imbalance"] >= 0.005,
            improvements["right_support_body_motion"] >= 0.005,
            improvements["phase_body_imbalance"] >= 0.005,
            improvements["actor_symmetry_error"] >= 0.002,
        )
    )
    style_gain = baseline["style_cost"] - candidate["style_cost"]
    minimum_style_gain = max(
        cfg["minimum_style_gain_abs"],
        cfg["minimum_style_gain_rel"] * baseline["style_cost"],
    )
    if material_improvements < cfg["material_min_required"]:
        soft_reasons.append(
            "only {} target gait metric(s) improved materially".format(
                material_improvements
            )
        )
    if style_gain < minimum_style_gain:
        soft_reasons.append(
            "style gain {:.4f} is below {:.4f}".format(
                style_gain, minimum_style_gain
            )
        )

    performance_delta = (
        cfg["weight_completion"]
        * (candidate["completion"] - baseline["completion"])
        - cfg["weight_fall"] * (candidate["fall"] - baseline["fall"])
        - cfg["weight_path"] * (
            candidate["path_failure"] - baseline["path_failure"]
        )
    )
    selection_score = style_gain + 0.20 * performance_delta
    if cfg["reject_on_negative_performance"] and performance_delta < 0:
        hard_reasons.append(
            "performance delta is negative ({:+.4f})".format(
                performance_delta
            )
        )
    fallback_eligible = (
        cfg["allow_safe_fallback"]
        and not hard_reasons
        and style_gain >= cfg["fallback_min_style_gain"]
        and selection_score >= cfg["fallback_min_score"]
        and material_improvements >= cfg["fallback_material_min"]
    )
    reasons = hard_reasons + soft_reasons
    return {
        "eligible": not reasons,
        "hard_safe": not hard_reasons,
        "fallback_eligible": fallback_eligible,
        "reasons": reasons,
        "hard_reasons": hard_reasons,
        "soft_reasons": soft_reasons,
        "baseline": baseline,
        "candidate": candidate,
        "improvements": improvements,
        "material_improvements": material_improvements,
        "high_target_baseline": high_target_baseline,
        "high_target_candidate": high_target_candidate,
        "target_deltas": target_deltas,
        "target_group_deltas": target_group_deltas,
        "target_group_improvements": target_group_improvements,
        "target_groups_improved": target_groups_improved,
        "style_gain": style_gain,
        "performance_delta": performance_delta,
        "selection_score": selection_score,
        "high_level_episode_tolerance": high_tolerance,
        "selection_mode": cfg["mode"],
    }


def parse_candidate(value):
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "--candidate must use NAME|EVALUATION.csv|CHECKPOINT.pt"
        )
    return {
        "name": parts[0],
        "evaluation": os.path.abspath(parts[1]),
        "checkpoint": os.path.abspath(parts[2]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.candidate:
        raise ValueError("at least one --candidate is required")
    if args.episodes is not None and args.episodes < 1:
        raise ValueError("--episodes must be positive")
    baseline_path = os.path.abspath(args.baseline)
    baseline_checkpoint = os.path.abspath(args.baseline_checkpoint)
    if not os.path.isfile(baseline_checkpoint):
        raise ValueError(
            "baseline checkpoint does not exist: " + baseline_checkpoint
        )
    baseline_rows = load_evaluation(baseline_path)
    candidates = []
    for raw_candidate in args.candidate:
        candidate = parse_candidate(raw_candidate)
        if not os.path.isfile(candidate["checkpoint"]):
            raise ValueError(
                "candidate checkpoint does not exist: "
                + candidate["checkpoint"]
            )
        result = compare(
            baseline_rows,
            load_evaluation(candidate["evaluation"]),
            episodes=args.episodes,
        )
        candidate["result"] = result
        candidates.append(candidate)
        print(
            "ISAAC_STABILITY_CANDIDATE name={} eligible={} hard_safe={} "
            "fallback={} style_gain={:+.4f} performance={:+.4f} "
            "score={:+.4f} target_groups={}/3".format(
                candidate["name"],
                result["eligible"],
                result["hard_safe"],
                result["fallback_eligible"],
                result["style_gain"],
                result["performance_delta"],
                result["selection_score"],
                result["target_groups_improved"],
            )
        )
        print(
            "ISAAC_STABILITY_TARGET_DELTAS name={} lateral={:+.4f} "
            "stride={:+.4f} right_support_shake={:+.4f}".format(
                candidate["name"],
                result["target_group_deltas"]["lateral"],
                result["target_group_deltas"]["stride"],
                result["target_group_deltas"]["right_support_shake"],
            )
        )
        for reason in result["reasons"]:
            print(
                "ISAAC_STABILITY_CANDIDATE_REJECT name={} reason={}".format(
                    candidate["name"], reason
                )
            )

    eligible = [
        candidate for candidate in candidates
        if candidate["result"]["eligible"]
    ]
    if eligible:
        winner = max(
            eligible,
            key=lambda candidate: candidate["result"]["selection_score"],
        )
        selection_kind = "eligible"
        winner_name = winner["name"]
        winner_checkpoint = winner["checkpoint"]
        winner_evaluation = winner["evaluation"]
        improved = True
    else:
        fallback = [
            candidate for candidate in candidates
            if candidate["result"]["fallback_eligible"]
        ]
        if fallback:
            winner = max(
                fallback,
                key=lambda candidate: candidate["result"]["selection_score"],
            )
            selection_kind = "safe_fallback"
            winner_name = winner["name"]
            winner_checkpoint = winner["checkpoint"]
            winner_evaluation = winner["evaluation"]
            improved = True
            print(
                "ISAAC_STABILITY_SAFE_FALLBACK name={} score={:+.4f} "
                "style_gain={:+.4f}".format(
                    winner_name,
                    winner["result"]["selection_score"],
                    winner["result"]["style_gain"],
                )
            )
        else:
            selection_kind = "baseline"
            winner_name = "baseline"
            winner_checkpoint = baseline_checkpoint
            winner_evaluation = baseline_path
            improved = False

    decision = {
        "improved": improved,
        "selection_kind": selection_kind,
        "selection_mode": compare_config()["mode"],
        "winner": {
            "name": winner_name,
            "checkpoint": winner_checkpoint,
            "evaluation": winner_evaluation,
        },
        "baseline": {
            "checkpoint": baseline_checkpoint,
            "evaluation": baseline_path,
        },
        "candidates": candidates,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as json_file:
        json.dump(decision, json_file, indent=2)
    print(
        "ISAAC_STABILITY_WINNER name={} improved={} kind={} "
        "checkpoint={}".format(
            winner_name, improved, selection_kind, winner_checkpoint
        )
    )
    print("ISAAC_STABILITY_DECISION={}".format(output_path))


if __name__ == "__main__":
    main()
