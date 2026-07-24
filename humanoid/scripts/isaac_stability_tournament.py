"""Choose a smoother Isaac stair policy without sacrificing climbing.

The tournament compares complete 0--10 cm evaluation tables generated from
the same deterministic seed.  Completion/fall checks are expressed in whole
episode resolution, while selection is driven by action motion, left/right
swing balance, and lateral drift.  A candidate must improve at least two of
those style dimensions and pass every safety guard.
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
        + 4.0 * stride
        + 2.0 * signed_lateral
        + max_lateral
        + 0.25 * yaw
        + 0.5 * double_flight
        + 8.0 * right_inward
        + 4.0 * left_inward
        + 0.5 * phase_motion_imbalance
        + 0.5 * left_swing_body_motion
        + 0.5 * phase_body_imbalance
        + actor_symmetry_error
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
    baseline = aggregate(baseline_rows)
    candidate = aggregate(candidate_rows)
    if episodes is None:
        episodes = min(
            int(float(baseline_rows[level].get("episodes", 1)))
            for level in LEVELS
        )
    reasons = []

    high_baseline = baseline_rows[4]
    high_candidate = candidate_rows[4]
    high_tolerance = episode_tolerance(
        high_baseline, high_candidate, 0.025, episodes
    )
    if high_candidate["completion_rate"] < (
        high_baseline["completion_rate"] - high_tolerance
    ):
        reasons.append(
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
        reasons.append(
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
        reasons.append(
            "10 cm path failure increased {:.3%} -> {:.3%}".format(
                high_baseline["path_failure_rate"],
                high_candidate["path_failure_rate"],
            )
        )

    for level in LEVELS:
        level_baseline = baseline_rows[level]
        level_candidate = candidate_rows[level]
        level_tolerance = episode_tolerance(
            level_baseline, level_candidate, 0.08, episodes
        )
        if level_candidate["completion_rate"] < (
            level_baseline["completion_rate"] - level_tolerance
        ):
            reasons.append(
                "level {} completion regressed {:.3%} -> {:.3%}".format(
                    level,
                    level_baseline["completion_rate"],
                    level_candidate["completion_rate"],
                )
            )
        if level_candidate["fall_rate"] > (
            level_baseline["fall_rate"] + level_tolerance
        ):
            reasons.append(
                "level {} fall increased {:.3%} -> {:.3%}".format(
                    level,
                    level_baseline["fall_rate"],
                    level_candidate["fall_rate"],
                )
            )

    if candidate["completion"] < baseline["completion"] - 0.025:
        reasons.append(
            "weighted completion regressed {:.3%} -> {:.3%}".format(
                baseline["completion"], candidate["completion"]
            )
        )
    if candidate["fall"] > baseline["fall"] + 0.025:
        reasons.append(
            "weighted fall increased {:.3%} -> {:.3%}".format(
                baseline["fall"], candidate["fall"]
            )
        )
    if candidate["path_failure"] > baseline["path_failure"] + 0.025:
        reasons.append(
            "weighted path failure increased {:.3%} -> {:.3%}".format(
                baseline["path_failure"], candidate["path_failure"]
            )
        )
    if candidate["command_error"] > baseline["command_error"] + 0.02:
        reasons.append(
            "weighted command error increased {:.4f} -> {:.4f}".format(
                baseline["command_error"], candidate["command_error"]
            )
        )
    if candidate["max_lateral"] > baseline["max_lateral"] + 0.008:
        reasons.append(
            "weighted lateral deviation increased {:.4f} -> {:.4f}".format(
                baseline["max_lateral"], candidate["max_lateral"]
            )
        )
    if candidate["yaw"] > baseline["yaw"] + 0.02:
        reasons.append(
            "weighted yaw deviation increased {:.4f} -> {:.4f}".format(
                baseline["yaw"], candidate["yaw"]
            )
        )
    if candidate["action_motion"] > (
        baseline["action_motion"] * 1.02 + 0.005
    ):
        reasons.append(
            "action motion increased {:.4f} -> {:.4f}".format(
                baseline["action_motion"], candidate["action_motion"]
            )
        )
    if candidate["stride_imbalance"] > (
        baseline["stride_imbalance"] + 0.008
    ):
        reasons.append(
            "stride imbalance increased {:.4f} -> {:.4f}".format(
                baseline["stride_imbalance"],
                candidate["stride_imbalance"],
            )
        )
    if candidate["signed_lateral"] > baseline["signed_lateral"] + 0.008:
        reasons.append(
            "signed lateral error increased {:.4f} -> {:.4f}".format(
                baseline["signed_lateral"], candidate["signed_lateral"]
            )
        )
    if candidate["right_foot_inward"] > (
        baseline["right_foot_inward"] + 0.004
    ):
        reasons.append(
            "right-foot inward error increased {:.4f} -> {:.4f}".format(
                baseline["right_foot_inward"],
                candidate["right_foot_inward"],
            )
        )
    if candidate["left_swing_action_motion"] > (
        baseline["left_swing_action_motion"] * 1.03 + 0.005
    ):
        reasons.append(
            "right-support/left-swing action motion increased "
            "{:.4f} -> {:.4f}".format(
                baseline["left_swing_action_motion"],
                candidate["left_swing_action_motion"],
            )
        )
    if candidate["left_swing_body_motion"] > (
        baseline["left_swing_body_motion"] * 1.05 + 0.005
    ):
        reasons.append(
            "right-support/left-swing body motion increased "
            "{:.4f} -> {:.4f}".format(
                baseline["left_swing_body_motion"],
                candidate["left_swing_body_motion"],
            )
        )
    if candidate["actor_symmetry_error"] > (
        baseline["actor_symmetry_error"] * 1.05 + 0.002
    ):
        reasons.append(
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
        "right_foot_inward": (
            baseline["right_foot_inward"]
            - candidate["right_foot_inward"]
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
            improvements["right_foot_inward"] >= 0.001,
            improvements["right_support_motion"] >= 0.005,
            improvements["phase_action_imbalance"] >= 0.005,
            improvements["right_support_body_motion"] >= 0.005,
            improvements["phase_body_imbalance"] >= 0.005,
            improvements["actor_symmetry_error"] >= 0.002,
        )
    )
    style_gain = baseline["style_cost"] - candidate["style_cost"]
    minimum_style_gain = max(0.01, 0.01 * baseline["style_cost"])
    if material_improvements < 2:
        reasons.append(
            "only {} target gait metric(s) improved materially".format(
                material_improvements
            )
        )
    if style_gain < minimum_style_gain:
        reasons.append(
            "style gain {:.4f} is below {:.4f}".format(
                style_gain, minimum_style_gain
            )
        )

    performance_delta = (
        4.0 * (candidate["completion"] - baseline["completion"])
        - 3.0 * (candidate["fall"] - baseline["fall"])
        - 2.0 * (
            candidate["path_failure"] - baseline["path_failure"]
        )
    )
    selection_score = style_gain + 0.20 * performance_delta
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "baseline": baseline,
        "candidate": candidate,
        "improvements": improvements,
        "material_improvements": material_improvements,
        "style_gain": style_gain,
        "performance_delta": performance_delta,
        "selection_score": selection_score,
        "high_level_episode_tolerance": high_tolerance,
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
            "ISAAC_STABILITY_CANDIDATE name={} eligible={} "
            "style_gain={:+.4f} performance={:+.4f} score={:+.4f}".format(
                candidate["name"],
                result["eligible"],
                result["style_gain"],
                result["performance_delta"],
                result["selection_score"],
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
        winner_name = winner["name"]
        winner_checkpoint = winner["checkpoint"]
        winner_evaluation = winner["evaluation"]
        improved = True
    else:
        winner_name = "baseline"
        winner_checkpoint = baseline_checkpoint
        winner_evaluation = baseline_path
        improved = False

    decision = {
        "improved": improved,
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
        "ISAAC_STABILITY_WINNER name={} improved={} checkpoint={}".format(
            winner_name, improved, winner_checkpoint
        )
    )
    print("ISAAC_STABILITY_DECISION={}".format(output_path))


if __name__ == "__main__":
    main()
