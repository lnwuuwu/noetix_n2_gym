"""Select a staged Isaac Gym gait candidate without sacrificing 10 cm climbing."""

import argparse
import csv
import json
import math
import os


MIN_COMPLETION = (0.96, 0.95, 0.94, 0.92, 0.90)
MAX_FALL = (0.05, 0.06, 0.07, 0.08, 0.10)
MAX_LATERAL = (0.060, 0.065, 0.075, 0.085, 0.100)
MAX_YAW = (0.20, 0.21, 0.23, 0.25, 0.28)


def load_level(path, level):
    with open(path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    matches = [
        row for row in rows if int(float(row["terrain_level"])) == int(level)
    ]
    if len(matches) != 1:
        raise ValueError(
            "{} must contain exactly one row for level {}, found {}".format(
                path, level, len(matches)
            )
        )
    result = {}
    for key, value in matches[0].items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            result[key] = value
            continue
        if not math.isfinite(numeric):
            raise ValueError("{} has non-finite {}".format(path, key))
        result[key] = numeric
    return result


def stride_imbalance(summary):
    return abs(
        summary["mean_left_swing_length_m"]
        - summary["mean_right_swing_length_m"]
    )


def action_motion(summary):
    return (
        summary["mean_action_rate_rms"]
        + 0.5 * summary["mean_action_accel_rms"]
    )


def stability_score(summary):
    """Higher is better; all terms use deterministic episode summaries."""
    return (
        8.0 * summary["completion_rate"]
        - 6.0 * summary["fall_rate"]
        - 4.0 * summary["path_failure_rate"]
        - 4.0 * summary["mean_max_lateral_deviation_m"]
        - 1.5 * summary["mean_max_yaw_deviation_rad"]
        - 2.0 * abs(summary["mean_final_lateral_position_m"])
        - 3.0 * stride_imbalance(summary)
        - 2.0 * action_motion(summary)
        - 1.0 * summary["mean_double_flight_fraction"]
        - 2.0 * summary["mean_command_error_m_s"]
    )


def _check_at_most(reasons, name, candidate, limit):
    if candidate > limit:
        reasons.append(
            "{} {:.6f} exceeds {:.6f}".format(name, candidate, limit)
        )


def decide(baseline_stage, candidate_stage, baseline_high, candidate_high, level):
    if not 0 <= level < len(MIN_COMPLETION):
        raise ValueError("stage level must be in [0, 4]")
    reasons = []

    required_completion = min(
        MIN_COMPLETION[level],
        max(0.70, baseline_stage["completion_rate"] - 0.03),
    )
    if candidate_stage["completion_rate"] < required_completion:
        reasons.append(
            "stage completion {:.3%} is below {:.3%}".format(
                candidate_stage["completion_rate"], required_completion
            )
        )
    _check_at_most(
        reasons,
        "stage fall",
        candidate_stage["fall_rate"],
        max(MAX_FALL[level], baseline_stage["fall_rate"] + 0.03),
    )
    _check_at_most(
        reasons,
        "stage lateral",
        candidate_stage["mean_max_lateral_deviation_m"],
        max(
            MAX_LATERAL[level],
            baseline_stage["mean_max_lateral_deviation_m"] + 0.008,
        ),
    )
    _check_at_most(
        reasons,
        "stage yaw",
        candidate_stage["mean_max_yaw_deviation_rad"],
        max(
            MAX_YAW[level],
            baseline_stage["mean_max_yaw_deviation_rad"] + 0.015,
        ),
    )
    _check_at_most(
        reasons,
        "stage signed lateral",
        abs(candidate_stage["mean_final_lateral_position_m"]),
        max(
            0.05,
            abs(baseline_stage["mean_final_lateral_position_m"]) + 0.01,
        ),
    )
    _check_at_most(
        reasons,
        "stage stride imbalance",
        stride_imbalance(candidate_stage),
        max(0.04, stride_imbalance(baseline_stage) + 0.01),
    )
    _check_at_most(
        reasons,
        "stage action motion",
        action_motion(candidate_stage),
        action_motion(baseline_stage) * 1.08 + 0.003,
    )

    if candidate_high["completion_rate"] < (
        baseline_high["completion_rate"] - 0.03
    ):
        reasons.append(
            "10 cm completion regressed {:.3%} -> {:.3%}".format(
                baseline_high["completion_rate"],
                candidate_high["completion_rate"],
            )
        )
    _check_at_most(
        reasons,
        "10 cm fall",
        candidate_high["fall_rate"],
        baseline_high["fall_rate"] + 0.03,
    )
    _check_at_most(
        reasons,
        "10 cm lateral",
        candidate_high["mean_max_lateral_deviation_m"],
        baseline_high["mean_max_lateral_deviation_m"] + 0.01,
    )
    _check_at_most(
        reasons,
        "10 cm stride imbalance",
        stride_imbalance(candidate_high),
        max(0.04, stride_imbalance(baseline_high) + 0.01),
    )
    _check_at_most(
        reasons,
        "10 cm action motion",
        action_motion(candidate_high),
        action_motion(baseline_high) * 1.10 + 0.005,
    )

    baseline_stage_score = stability_score(baseline_stage)
    candidate_stage_score = stability_score(candidate_stage)
    baseline_high_score = stability_score(baseline_high)
    candidate_high_score = stability_score(candidate_high)
    if candidate_stage_score < baseline_stage_score:
        reasons.append(
            "stage stability score regressed {:.4f} -> {:.4f}".format(
                baseline_stage_score, candidate_stage_score
            )
        )
    if candidate_high_score < baseline_high_score - 0.10:
        reasons.append(
            "10 cm stability score regressed {:.4f} -> {:.4f}".format(
                baseline_high_score, candidate_high_score
            )
        )

    return {
        "accepted": not reasons,
        "stage_level": level,
        "reasons": reasons,
        "scores": {
            "baseline_stage": baseline_stage_score,
            "candidate_stage": candidate_stage_score,
            "baseline_10cm": baseline_high_score,
            "candidate_10cm": candidate_high_score,
        },
        "derived": {
            "baseline_stage_stride_imbalance_m": stride_imbalance(
                baseline_stage
            ),
            "candidate_stage_stride_imbalance_m": stride_imbalance(
                candidate_stage
            ),
            "baseline_stage_action_motion": action_motion(baseline_stage),
            "candidate_stage_action_motion": action_motion(candidate_stage),
            "baseline_10cm_stride_imbalance_m": stride_imbalance(
                baseline_high
            ),
            "candidate_10cm_stride_imbalance_m": stride_imbalance(
                candidate_high
            ),
            "baseline_10cm_action_motion": action_motion(baseline_high),
            "candidate_10cm_action_motion": action_motion(candidate_high),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-stage", required=True)
    parser.add_argument("--candidate-stage", required=True)
    parser.add_argument("--baseline-high", required=True)
    parser.add_argument("--candidate-high", required=True)
    parser.add_argument("--stage-level", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    decision = decide(
        load_level(args.baseline_stage, args.stage_level),
        load_level(args.candidate_stage, args.stage_level),
        load_level(args.baseline_high, 4),
        load_level(args.candidate_high, 4),
        args.stage_level,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as json_file:
        json.dump(decision, json_file, indent=2)
    print(
        "ISAAC_STABILITY_GATE level={} accepted={} "
        "stage={:.4f}->{:.4f} 10cm={:.4f}->{:.4f}".format(
            args.stage_level,
            decision["accepted"],
            decision["scores"]["baseline_stage"],
            decision["scores"]["candidate_stage"],
            decision["scores"]["baseline_10cm"],
            decision["scores"]["candidate_10cm"],
        )
    )
    for reason in decision["reasons"]:
        print("ISAAC_STABILITY_REJECT_REASON " + reason)
    return 0 if decision["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
