"""Select a FastStair checkpoint with absolute and relative safety gates.

The script is Isaac-Gym independent.  It consumes the five-row CSV files
written by ``eval_stairs.py``, so checkpoint screening and final holdout
approval can run as a small deterministic post-processing step.
"""

import argparse
import csv
import json
import math
import os


LEVELS = tuple(range(5))
LEVEL_WEIGHTS = {
    0: 0.10,
    1: 0.15,
    2: 0.15,
    3: 0.20,
    4: 0.40,
}


def load_evaluation(path):
    with open(path, newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    rows = {}
    for source in source_rows:
        level = int(float(source["terrain_level"]))
        if level in rows:
            raise ValueError(
                "{} contains duplicate level {}".format(path, level)
            )
        row = {}
        for key, value in source.items():
            try:
                converted = float(value)
            except (TypeError, ValueError):
                row[key] = value
                continue
            if not math.isfinite(converted):
                raise ValueError(
                    "{} contains non-finite {}".format(path, key)
                )
            row[key] = converted
        rows[level] = row
    missing = [level for level in LEVELS if level not in rows]
    if missing:
        raise ValueError(
            "{} is missing levels {}".format(
                path, ",".join(str(level) for level in missing)
            )
        )
    return rows


def _optional(row, key, default=0.0):
    return float(row.get(key, default))


def _weighted(rows, key, default=0.0):
    return sum(
        LEVEL_WEIGHTS[level] * _optional(rows[level], key, default)
        for level in LEVELS
    )


def aggregate(rows):
    action_motion = sum(
        LEVEL_WEIGHTS[level]
        * (
            _optional(rows[level], "mean_action_rate_rms")
            + 0.5 * _optional(rows[level], "mean_action_accel_rms")
        )
        for level in LEVELS
    )
    return {
        "completion": _weighted(rows, "completion_rate"),
        "fall": _weighted(rows, "fall_rate"),
        "path": _weighted(rows, "path_failure_rate"),
        "lateral": _weighted(rows, "mean_max_lateral_deviation_m"),
        "yaw": _weighted(rows, "mean_max_yaw_deviation_rad"),
        "command_error": _weighted(rows, "mean_command_error_m_s"),
        "action_motion": action_motion,
        "planner_valid": _weighted(
            rows, "mean_faststair_planner_valid_fraction"
        ),
        "foothold_error": _weighted(
            rows, "mean_faststair_foothold_error_m"
        ),
        "edge_margin": _weighted(
            rows, "mean_faststair_edge_margin_m"
        ),
    }


def score(rows):
    metrics = aggregate(rows)
    level4 = rows[4]
    return (
        5.0 * metrics["completion"]
        + 2.0 * float(level4["completion_rate"])
        - 4.0 * metrics["fall"]
        - 2.0 * metrics["path"]
        - 2.0 * metrics["lateral"] / 0.10
        - 0.35 * metrics["action_motion"]
        - 2.0 * metrics["foothold_error"] / 0.08
        + 0.50 * metrics["planner_valid"]
        + 0.25 * min(metrics["edge_margin"] / 0.02, 1.0)
    )


def absolute_gate(rows):
    """Return reasons that prevent a model from becoming a saved best."""
    metrics = aggregate(rows)
    reasons = []
    checks = (
        (
            float(rows[4]["completion_rate"]) < 0.90,
            "10 cm completion is below 90%",
        ),
        (
            float(rows[4]["fall_rate"]) > 0.05,
            "10 cm fall exceeds 5%",
        ),
        (
            float(rows[4]["path_failure_rate"]) > 0.08,
            "10 cm path failure exceeds 8%",
        ),
        (
            float(rows[3]["completion_rate"]) < 0.85,
            "8 cm completion is below 85%",
        ),
        (
            float(rows[2]["completion_rate"]) < 0.65,
            "6 cm completion is below 65%",
        ),
        (
            metrics["completion"] < 0.72,
            "weighted completion is below 72%",
        ),
        (
            metrics["fall"] > 0.18,
            "weighted fall exceeds 18%",
        ),
        (
            metrics["path"] > 0.10,
            "weighted path failure exceeds 10%",
        ),
        (
            metrics["lateral"] > 0.12,
            "weighted lateral deviation exceeds 0.12 m",
        ),
        (
            metrics["action_motion"] > 1.20,
            "weighted action motion exceeds 1.20",
        ),
        (
            metrics["planner_valid"] < 0.90,
            "planner validity is below 90%",
        ),
        (
            metrics["foothold_error"] > 0.10,
            "planned touchdown error exceeds 0.10 m",
        ),
        (
            metrics["edge_margin"] < 0.005,
            "planned foothold edge margin is below 5 mm",
        ),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    return reasons


def relative_gate(baseline_rows, candidate_rows):
    """Prevent a new architecture from erasing the approved PPO baseline."""
    baseline = aggregate(baseline_rows)
    candidate = aggregate(candidate_rows)
    reasons = []
    comparisons = (
        (
            float(candidate_rows[4]["completion_rate"]) + 0.03
            < float(baseline_rows[4]["completion_rate"]),
            "10 cm completion regressed by more than 3%",
        ),
        (
            float(candidate_rows[4]["fall_rate"])
            > float(baseline_rows[4]["fall_rate"]) + 0.03,
            "10 cm fall increased by more than 3%",
        ),
        (
            float(candidate_rows[4]["path_failure_rate"])
            > float(baseline_rows[4]["path_failure_rate"]) + 0.03,
            "10 cm path failure increased by more than 3%",
        ),
        (
            candidate["completion"] + 0.05 < baseline["completion"],
            "weighted completion regressed by more than 5%",
        ),
        (
            candidate["lateral"] > baseline["lateral"] + 0.01,
            "weighted lateral deviation increased by more than 1 cm",
        ),
        (
            candidate["action_motion"]
            > 1.08 * baseline["action_motion"] + 0.01,
            "weighted action motion increased by more than 8%",
        ),
    )
    reasons.extend(reason for failed, reason in comparisons if failed)
    return reasons


def parse_candidate(value):
    fields = value.split("|", 2)
    if len(fields) != 3 or not all(fields):
        raise ValueError(
            "Candidate must use name|evaluation.csv|model.pt syntax"
        )
    name, evaluation, checkpoint = fields
    if not os.path.isfile(evaluation):
        raise FileNotFoundError(evaluation)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    return {
        "name": name,
        "evaluation": os.path.abspath(evaluation),
        "checkpoint": os.path.abspath(checkpoint),
    }


def select(args):
    baseline_rows = (
        load_evaluation(args.baseline) if args.baseline else None
    )
    results = []
    for value in args.candidate:
        candidate = parse_candidate(value)
        rows = load_evaluation(candidate["evaluation"])
        reasons = absolute_gate(rows)
        if baseline_rows is not None:
            reasons.extend(relative_gate(baseline_rows, rows))
        metrics = aggregate(rows)
        candidate.update(
            {
                "score": score(rows),
                "eligible": not reasons,
                "reasons": reasons,
                "metrics": metrics,
            }
        )
        results.append(candidate)
        print(
            "FASTSTAIR_CANDIDATE name={} eligible={} score={:+.4f} "
            "completion={:.1%} fall={:.1%} path={:.1%} lateral={:.3f}m "
            "plan_valid={:.1%} plan_error={:.3f}m edge={:.3f}m".format(
                candidate["name"],
                candidate["eligible"],
                candidate["score"],
                metrics["completion"],
                metrics["fall"],
                metrics["path"],
                metrics["lateral"],
                metrics["planner_valid"],
                metrics["foothold_error"],
                metrics["edge_margin"],
            )
        )
        for reason in reasons:
            print(
                "FASTSTAIR_CANDIDATE_REJECT name={} reason={}".format(
                    candidate["name"], reason
                )
            )

    eligible = [candidate for candidate in results if candidate["eligible"]]
    winner = max(eligible, key=lambda item: item["score"]) if eligible else None
    screen_best = max(results, key=lambda item: item["score"])
    if winner is None:
        print(
            "FASTSTAIR_WINNER name=NONE approved=False "
            "screen_best={}".format(screen_best["name"])
        )
    else:
        print(
            "FASTSTAIR_WINNER name={} approved=True checkpoint={}".format(
                winner["name"], winner["checkpoint"]
            )
        )
    decision = {
        "approved": winner is not None,
        "winner": winner,
        "screen_best": screen_best,
        "baseline": (
            os.path.abspath(args.baseline) if args.baseline else None
        ),
        "baseline_checkpoint": (
            os.path.abspath(args.baseline_checkpoint)
            if args.baseline_checkpoint
            else None
        ),
        "candidates": results,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(decision, stream, indent=2)
    print("FASTSTAIR_DECISION={}".format(output))
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Repeat name|evaluation.csv|model.pt for each checkpoint.",
    )
    parser.add_argument("--output", required=True)
    select(parser.parse_args())


if __name__ == "__main__":
    main()
