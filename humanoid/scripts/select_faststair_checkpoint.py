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


def _advance_imbalance(row):
    """Return normalized left/right tread-count imbalance in ``[0, 1]``."""
    left = _optional(row, "mean_left_tread_advances")
    right = _optional(row, "mean_right_tread_advances")
    return abs(left - right) / max(left + right, 1.0e-6)


def _swing_length_imbalance(row):
    return abs(
        _optional(row, "mean_left_swing_length_m")
        - _optional(row, "mean_right_swing_length_m")
    )


def _right_support_shake(row):
    """Measure the reported right-support/left-swing instability.

    ``eval_stairs.py`` labels phase metrics by the swinging foot.  The user's
    visible defect happens while the right foot supports and the left swings,
    hence the left-swing acceleration and roll-rate channels below.
    """
    return max(
        _optional(row, "mean_left_swing_action_accel_rms"),
        _optional(row, "mean_left_swing_roll_rate_rms"),
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
        "success": _weighted(rows, "success_rate"),
        "curriculum_completion": _weighted(
            rows, "curriculum_completion_rate"
        ),
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
        "alternating_count": _weighted(
            rows, "mean_alternating_tread_count"
        ),
        "alternating_rate": _weighted(
            rows, "mean_alternating_tread_rate"
        ),
        "same_tread_join": _weighted(
            rows, "mean_same_tread_join_rate"
        ),
        "repeated_lead": _weighted(
            rows, "mean_repeated_lead_rate"
        ),
        "phase_contact": _weighted(rows, "mean_phase_contact_match"),
        "sagittal_phase": _weighted(
            rows, "mean_sagittal_foot_phase_match"
        ),
        "advance_imbalance": sum(
            LEVEL_WEIGHTS[level] * _advance_imbalance(rows[level])
            for level in LEVELS
        ),
        "swing_imbalance": sum(
            LEVEL_WEIGHTS[level] * _swing_length_imbalance(rows[level])
            for level in LEVELS
        ),
        "right_support_shake": sum(
            LEVEL_WEIGHTS[level] * _right_support_shake(rows[level])
            for level in LEVELS
        ),
    }


def _mean_levels(rows, levels, key, default=0.0):
    return sum(
        _optional(rows[level], key, default) for level in levels
    ) / float(len(levels))


def _mean_derived(rows, levels, function):
    return sum(function(rows[level]) for level in levels) / float(len(levels))


def _gait_score(rows, levels):
    """Dense ranking term for natural, balanced, non-shaking stair gait."""
    alternating_count = _mean_levels(
        rows, levels, "mean_alternating_tread_count"
    )
    alternating_rate = _mean_levels(
        rows, levels, "mean_alternating_tread_rate"
    )
    same_tread_join = _mean_levels(
        rows, levels, "mean_same_tread_join_rate"
    )
    repeated_lead = _mean_levels(
        rows, levels, "mean_repeated_lead_rate"
    )
    advance_imbalance = _mean_derived(rows, levels, _advance_imbalance)
    swing_imbalance = _mean_derived(
        rows, levels, _swing_length_imbalance
    )
    support_shake = _mean_derived(rows, levels, _right_support_shake)
    return (
        1.50 * min(alternating_count / 3.0, 1.0)
        + 1.00 * alternating_rate
        - 1.25 * same_tread_join
        - 0.75 * repeated_lead
        - 1.25 * advance_imbalance
        - 0.75 * swing_imbalance / 0.10
        - 0.25 * support_shake
    )


def score(rows, stage=0):
    """Return a stage-aware ranking score.

    Early stages deliberately rank the terrain rows that were trained instead
    of allowing untrained 8/10 cm rows to dominate checkpoint selection.
    """
    if stage == 1:
        row = rows[0]
        action_motion = (
            _optional(row, "mean_action_rate_rms")
            + 0.5 * _optional(row, "mean_action_accel_rms")
        )
        return (
            6.0 * _optional(row, "completion_rate")
            + 1.0 * _optional(row, "first_step_rate")
            + 1.0 * _optional(row, "curriculum_completion_rate")
            - 4.0 * _optional(row, "fall_rate")
            - 2.0 * _optional(row, "path_failure_rate")
            - 1.5
            * _optional(row, "mean_max_lateral_deviation_m")
            / 0.10
            - 0.25 * action_motion
            + 0.75
            * _optional(
                row, "mean_faststair_planner_valid_fraction"
            )
            + _gait_score(rows, (0,))
        )
    if stage == 2:
        levels = (0, 1, 2)
        completion = _mean_levels(rows, levels, "completion_rate")
        fall = _mean_levels(rows, levels, "fall_rate")
        path = _mean_levels(rows, levels, "path_failure_rate")
        lateral = _mean_levels(
            rows, levels, "mean_max_lateral_deviation_m"
        )
        planner_valid = _mean_levels(
            rows,
            levels,
            "mean_faststair_planner_valid_fraction",
        )
        return (
            5.0 * completion
            + 2.0 * _optional(rows[2], "completion_rate")
            + 1.0 * _optional(rows[2], "curriculum_completion_rate")
            - 4.0 * fall
            - 2.0 * path
            - 1.5 * lateral / 0.10
            + 0.75 * planner_valid
            + _gait_score(rows, levels)
        )
    metrics = aggregate(rows)
    level4 = rows[4]
    return (
        5.0 * metrics["completion"]
        + 2.0 * float(level4["completion_rate"])
        + 1.5 * metrics["success"]
        - 4.0 * metrics["fall"]
        - 2.0 * metrics["path"]
        - 2.0 * metrics["lateral"] / 0.10
        - 0.35 * metrics["action_motion"]
        - 2.0 * metrics["foothold_error"] / 0.08
        + 0.50 * metrics["planner_valid"]
        + 0.25 * min(metrics["edge_margin"] / 0.02, 1.0)
        + _gait_score(rows, LEVELS)
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
            _optional(rows[4], "success_rate") < 0.65,
            "10 cm natural-gait success is below 65%",
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
            _optional(rows[3], "success_rate") < 0.65,
            "8 cm natural-gait success is below 65%",
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
        (
            _mean_levels(
                rows, (3, 4), "mean_alternating_tread_count"
            )
            < 2.50,
            "8/10 cm alternating-tread count is below 2.50",
        ),
        (
            _mean_levels(
                rows, (3, 4), "mean_alternating_tread_rate"
            )
            < 0.40,
            "8/10 cm alternating-tread rate is below 40%",
        ),
        (
            _mean_levels(
                rows, (3, 4), "mean_same_tread_join_rate"
            )
            > 0.25,
            "8/10 cm same-tread join rate exceeds 25%",
        ),
        (
            _mean_levels(
                rows, (3, 4), "mean_repeated_lead_rate"
            )
            > 0.25,
            "8/10 cm repeated-lead rate exceeds 25%",
        ),
        (
            min(
                _optional(rows[level], "mean_left_tread_advances")
                for level in (3, 4)
            )
            < 2.50
            or min(
                _optional(rows[level], "mean_right_tread_advances")
                for level in (3, 4)
            )
            < 2.50,
            "both feet must average at least 2.50 advances on 8/10 cm",
        ),
        (
            _mean_derived(rows, (3, 4), _advance_imbalance) > 0.15,
            "8/10 cm left/right tread-advance imbalance exceeds 15%",
        ),
        (
            _mean_derived(rows, (3, 4), _swing_length_imbalance) > 0.08,
            "8/10 cm swing-length imbalance exceeds 0.08 m",
        ),
        (
            _mean_levels(
                rows, (3, 4), "mean_sagittal_foot_phase_match"
            )
            < 0.72,
            "8/10 cm sagittal foot-phase match is below 72%",
        ),
        (
            _mean_derived(rows, (3, 4), _right_support_shake) > 0.72,
            "8/10 cm right-support/left-swing shake exceeds 0.72",
        ),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    return reasons


def _stage_one_safety_gate(rows):
    """Basic migration/locomotion checks shared with bootstrap preflight."""
    row = rows[0]
    checks = (
        (
            _optional(row, "completion_rate") < 0.55,
            "stage 1: 2 cm completion is below 55%",
        ),
        (
            _optional(row, "first_step_rate") < 0.85,
            "stage 1: 2 cm first-step rate is below 85%",
        ),
        (
            _optional(row, "fall_rate") > 0.30,
            "stage 1: 2 cm fall exceeds 30%",
        ),
        (
            _optional(row, "path_failure_rate") > 0.25,
            "stage 1: 2 cm path failure exceeds 25%",
        ),
        (
            _optional(row, "mean_max_lateral_deviation_m") > 0.14,
            "stage 1: 2 cm lateral deviation exceeds 0.14 m",
        ),
        (
            _optional(
                row, "mean_faststair_planner_valid_fraction"
            )
            < 0.85,
            "stage 1: planner validity is below 85%",
        ),
        (
            _optional(row, "mean_faststair_edge_margin_m") < 0.003,
            "stage 1: foothold edge margin is below 3 mm",
        ),
    )
    return [reason for failed, reason in checks if failed]


def preflight_gate(rows, baseline_rows=None):
    """Validate a migrated Actor before spending time on PPO rollouts.

    Preflight intentionally checks only the 2 cm discovery row.  It is not a
    policy-approval gate: the untrained wider network cannot yet satisfy the
    natural-gait or all-height requirements.  A loose same-seed comparison
    still catches a broken observation migration without rejecting harmless
    small-batch differences between the two task configurations.
    """
    reasons = _stage_one_safety_gate(rows)
    if baseline_rows is None:
        return reasons
    baseline = baseline_rows[0]
    candidate = rows[0]
    comparisons = (
        (
            _optional(candidate, "completion_rate") + 0.12
            < _optional(baseline, "completion_rate"),
            "preflight: 2 cm completion regressed by more than 12%",
        ),
        (
            _optional(candidate, "fall_rate")
            > _optional(baseline, "fall_rate") + 0.12,
            "preflight: 2 cm fall increased by more than 12%",
        ),
        (
            _optional(candidate, "path_failure_rate")
            > _optional(baseline, "path_failure_rate") + 0.12,
            "preflight: 2 cm path failure increased by more than 12%",
        ),
    )
    reasons.extend(reason for failed, reason in comparisons if failed)
    return reasons


def stage_gate(rows, stage):
    """Return promotion blockers for an easy-to-hard training stage."""
    stage = int(stage)
    if stage == 3:
        return absolute_gate(rows)
    reasons = []
    if stage == 1:
        row = rows[0]
        reasons.extend(_stage_one_safety_gate(rows))
        checks = (
            (
                _optional(row, "mean_alternating_tread_count") < 1.00,
                "stage 1: 2 cm alternating-tread count is below 1.00",
            ),
            (
                _optional(row, "curriculum_completion_rate") < 0.20,
                "stage 1: natural-gait curriculum completion is below 20%",
            ),
            (
                _optional(row, "mean_alternating_tread_rate") < 0.15,
                "stage 1: 2 cm alternating-tread rate is below 15%",
            ),
            (
                min(
                    _optional(row, "mean_left_tread_advances"),
                    _optional(row, "mean_right_tread_advances"),
                )
                < 1.25,
                "stage 1: each foot must average at least 1.25 advances",
            ),
            (
                _advance_imbalance(row) > 0.40,
                "stage 1: left/right tread-advance imbalance exceeds 40%",
            ),
            (
                _optional(row, "mean_same_tread_join_rate") > 0.35,
                "stage 1: same-tread join rate exceeds 35%",
            ),
            (
                _swing_length_imbalance(row) > 0.14,
                "stage 1: swing-length imbalance exceeds 0.14 m",
            ),
            (
                _optional(row, "mean_sagittal_foot_phase_match") < 0.72,
                "stage 1: sagittal foot-phase match is below 72%",
            ),
            (
                _right_support_shake(row) > 0.62,
                "stage 1: right-support/left-swing shake exceeds 0.62",
            ),
        )
    elif stage == 2:
        levels = (0, 1, 2)
        checks = (
            (
                _optional(rows[2], "completion_rate") < 0.60,
                "stage 2: 6 cm completion is below 60%",
            ),
            (
                _mean_levels(rows, levels, "completion_rate") < 0.45,
                "stage 2: mean 2/4/6 cm completion is below 45%",
            ),
            (
                _optional(rows[1], "first_step_rate") < 0.65,
                "stage 2: 4 cm first-step rate is below 65%",
            ),
            (
                _optional(rows[2], "first_step_rate") < 0.90,
                "stage 2: 6 cm first-step rate is below 90%",
            ),
            (
                _mean_levels(rows, levels, "fall_rate") > 0.40,
                "stage 2: mean 2/4/6 cm fall exceeds 40%",
            ),
            (
                _mean_levels(rows, levels, "path_failure_rate") > 0.25,
                "stage 2: mean 2/4/6 cm path failure exceeds 25%",
            ),
            (
                _mean_levels(
                    rows,
                    levels,
                    "mean_faststair_planner_valid_fraction",
                )
                < 0.85,
                "stage 2: planner validity is below 85%",
            ),
            (
                _optional(
                    rows[2], "mean_alternating_tread_count"
                )
                < 1.50,
                "stage 2: 6 cm alternating-tread count is below 1.50",
            ),
            (
                _optional(rows[2], "curriculum_completion_rate") < 0.40,
                "stage 2: 6 cm natural-gait curriculum completion is below 40%",
            ),
            (
                _mean_levels(
                    rows, levels, "mean_alternating_tread_rate"
                )
                < 0.20,
                "stage 2: mean alternating-tread rate is below 20%",
            ),
            (
                min(
                    _optional(rows[2], "mean_left_tread_advances"),
                    _optional(rows[2], "mean_right_tread_advances"),
                )
                < 2.00,
                "stage 2: each foot must average 2.00 advances on 6 cm",
            ),
            (
                _advance_imbalance(rows[2]) > 0.25,
                "stage 2: 6 cm tread-advance imbalance exceeds 25%",
            ),
            (
                _mean_levels(
                    rows, levels, "mean_same_tread_join_rate"
                )
                > 0.32,
                "stage 2: mean same-tread join rate exceeds 32%",
            ),
            (
                _swing_length_imbalance(rows[2]) > 0.11,
                "stage 2: 6 cm swing-length imbalance exceeds 0.11 m",
            ),
            (
                _right_support_shake(rows[2]) > 0.66,
                "stage 2: 6 cm right-support/left-swing shake exceeds 0.66",
            ),
        )
    else:
        raise ValueError("FastStair stage must be 1, 2, or 3")
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
        if args.preflight:
            reasons = preflight_gate(rows, baseline_rows)
        elif args.stage:
            reasons = stage_gate(rows, args.stage)
        else:
            reasons = absolute_gate(rows)
        if baseline_rows is not None and not args.preflight:
            reasons.extend(relative_gate(baseline_rows, rows))
        metrics = aggregate(rows)
        candidate.update(
            {
                "score": score(rows, args.stage or (1 if args.preflight else 0)),
                "eligible": not reasons,
                "reasons": reasons,
                "metrics": metrics,
            }
        )
        results.append(candidate)
        print(
            "FASTSTAIR_CANDIDATE name={} eligible={} score={:+.4f} "
            "completion={:.1%} fall={:.1%} path={:.1%} lateral={:.3f}m "
            "plan_valid={:.1%} plan_error={:.3f}m edge={:.3f}m "
            "alternate={:.1%} join={:.1%} advance_imbalance={:.1%} "
            "swing_imbalance={:.3f}m support_shake={:.3f}".format(
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
                metrics["alternating_rate"],
                metrics["same_tread_join"],
                metrics["advance_imbalance"],
                metrics["swing_imbalance"],
                metrics["right_support_shake"],
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
        "stage": args.stage,
        "preflight": bool(args.preflight),
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
        "--preflight",
        action="store_true",
        help=(
            "Use the migration-only 2 cm gate before any PPO rollout. This "
            "does not approve a trained policy."
        ),
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2, 3),
        default=0,
        help=(
            "Use the promotion gate and ranking for an intermediate stage; "
            "omit for final holdout approval."
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Repeat name|evaluation.csv|model.pt for each checkpoint.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.preflight and args.stage:
        parser.error("--preflight and --stage are mutually exclusive")
    select(args)


if __name__ == "__main__":
    main()
