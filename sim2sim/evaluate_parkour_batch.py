#!/usr/bin/env python3
"""Evaluate several exported parkour policies headlessly in one fair batch."""

import argparse
import os
import time
from pathlib import Path

import yaml


# No keyboard or OpenGL window is needed for physics-only evaluation.
os.environ.setdefault("PYNPUT_BACKEND", "dummy")

from parkour_evaluation import (  # noqa: E402
    best_evaluation,
    evaluation_record,
    evaluation_report,
    write_evaluation_report,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _resolve_policy_path(path):
    expanded = str(path).replace(
        "{LEGGED_GYM_ROOT_DIR}", str(REPOSITORY_ROOT))
    candidate = Path(expanded).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run exported TorchScript policies on the same finite MuJoCo "
            "campaign and write one machine-readable JSON report."))
    parser.add_argument(
        "policy_paths", nargs="+",
        help="candidate policy_*.pt paths (shell globs are supported)")
    parser.add_argument(
        "--config_file", default="n2_parkour_slow_stable.yaml",
        help="YAML name under sim2sim/configs")
    parser.add_argument(
        "--duration", type=float, default=600.0,
        help="maximum simulated seconds per candidate (must be positive)")
    parser.add_argument(
        "--output", required=True,
        help="destination JSON report")
    parser.add_argument(
        "--command_x", type=float,
        help="override the YAML fallback command; campaign stage speeds remain "
             "controlled by campaign_command_speeds")
    parser.add_argument(
        "--course_mode", choices=("campaign", "up_only", "repeat", "single"),
        default="campaign",
        help="route state machine used for every candidate")
    parser.add_argument(
        "--verbose_progress", action="store_true",
        help="print the per-simulated-second status lines")
    parser.add_argument(
        "--action_filter_alpha", type=float,
        help="override deployment action filtering (1 means bypass)")
    parser.add_argument(
        "--action_gain", type=float,
        help="override deployment action amplitude")
    parser.add_argument(
        "--kd_scale", type=float,
        help="override deployment damping multiplier")
    return parser


def main():
    args = _parser().parse_args()
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive for batch evaluation")

    # Keep --help and report-helper tests usable on machines without MuJoCo.
    from sim2sim_parkour import cmd, run_mujoco

    config_path = (
        REPOSITORY_ROOT / "sim2sim/configs" / args.config_file)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    command_x = (
        float(args.command_x)
        if args.command_x is not None
        else float(config.get(
            "sim_command_x",
            config.get("ranges_lin_vel_x_max", 0.8))))

    policies = []
    seen = set()
    for raw_path in args.policy_paths:
        resolved = _resolve_policy_path(raw_path)
        if str(resolved) not in seen:
            policies.append(resolved)
            seen.add(str(resolved))

    records = []
    for candidate_index, policy_path in enumerate(policies, 1):
        print(
            f"\n[eval] candidate {candidate_index}/{len(policies)}: "
            f"{policy_path}")
        started = time.monotonic()
        try:
            if not policy_path.is_file():
                raise FileNotFoundError(
                    f"policy does not exist: {policy_path}")
            command = cmd()
            command.cmd[:] = [command_x, 0.0, 0.0]
            result = run_mujoco(
                args.config_file,
                command,
                goal_mode_override=config.get("goal_mode", "waypoints"),
                headless=True,
                duration_override=args.duration,
                policy_path_override=str(policy_path),
                debug_viz_override=False,
                course_mode_override=args.course_mode,
                quiet_progress=not args.verbose_progress,
                action_filter_alpha_override=args.action_filter_alpha,
                action_gain_override=args.action_gain,
                kd_scale_override=args.kd_scale)
            record = evaluation_record(
                policy_path, result=result,
                wall_time_seconds=time.monotonic() - started)
            print(
                "[eval] result: success=%s clears=%s/%s failures=%s "
                "resets=%s best_progress=%.3f max_x=%.2f sim_time=%.1fs"
                % (
                    record["success"],
                    record.get("stage_clears"),
                    record.get("stage_count"),
                    record.get("failures"),
                    record.get("resets"),
                    float(record.get("best_progress_fraction") or 0.0),
                    float(record.get("max_x") or 0.0),
                    float(record.get("sim_time") or 0.0)))
        except KeyboardInterrupt:
            raise
        except Exception as error:
            record = evaluation_record(
                policy_path, error=error,
                wall_time_seconds=time.monotonic() - started)
            print(
                f"[eval] ERROR {record['error_type']}: {record['error']}")
        records.append(record)

    control_overrides = {
        name: value for name, value in (
            ("action_filter_alpha", args.action_filter_alpha),
            ("action_gain", args.action_gain),
            ("kd_scale", args.kd_scale),
        ) if value is not None}
    report = evaluation_report(
        args.config_file, args.duration, records, command_x=command_x,
        control_overrides=control_overrides)
    output_path = _resolve_policy_path(args.output)
    write_evaluation_report(output_path, report)
    best = best_evaluation(records)
    print(f"\n[eval] JSON report: {output_path}")
    if best is None:
        print("[eval] no candidate completed a valid simulation")
        return 2
    print(
        "[eval] best: checkpoint=%s success=%s clears=%s/%s "
        "failures=%s resets=%s policy=%s"
        % (
            best.get("checkpoint"),
            best.get("success"),
            best.get("stage_clears"),
            best.get("stage_count"),
            best.get("failures"),
            best.get("resets"),
            best.get("policy_path")))
    return 0 if report["successful_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
