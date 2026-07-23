"""Rank the useful Isaac Gym stair checkpoints in one MuJoCo test.

The curated candidates represent the high-completion baseline, the first
natural curriculum policy, the tiered 10 cm policy, and the polished 10 cm
policy.  Missing runs are skipped explicitly, so this remains a one-command
tool on machines whose logs contain only part of the training history.
"""

import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = ROOT / "logs" / "n2_stairs_walk"

CURATED_CANDIDATES = (
    {
        "label": "phase_v2_5000",
        "iteration": 5000,
        "run_token": "stairs_walk_phase_v2_resume",
    },
    {
        "label": "natural_fast_8000",
        "iteration": 8000,
        "run_token": "natural_fast_curriculum_8000",
    },
    {
        "label": "tiered_8600",
        "iteration": 8600,
        "run_token": "tiered",
    },
    {
        "label": "natural_l4_9000",
        "iteration": 9000,
        "run_token": "natural_l4_polish_9000",
    },
)


def _default_output_dir():
    autodl = Path("/root/autodl-tmp/n2_eval")
    try:
        if autodl.parent.is_dir():
            return autodl / "isaac_mujoco_compare"
    except OSError:
        pass
    return ROOT / "reports" / "isaac_mujoco_compare"


def resolve_candidate(log_root, specification):
    pattern = str(
        Path(log_root)
        / "**"
        / "model_{}.pt".format(specification["iteration"])
    )
    matches = [
        Path(path)
        for path in glob.glob(pattern, recursive=True)
        if specification["run_token"] in str(Path(path).parent)
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _ranking_key(result):
    # Physical completion is the first gate.  When every transferred policy
    # fails it, prefer the one that remains inside the route longest with the
    # smallest heading divergence.
    return (
        -float(result["completion_rate"]),
        float(result["path_failure_rate"]),
        float(result["fall_rate"]),
        -float(result["mean_survival_time_s"]),
        float(result["mean_max_yaw_deviation_rad"]),
    )


def compare(args):
    log_root = Path(args.log_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ROOT / "sim2sim" / "eval_stairs_mujoco.py"

    candidates = []
    for specification in CURATED_CANDIDATES:
        checkpoint = resolve_candidate(log_root, specification)
        if checkpoint is None:
            print(
                "SKIP {}: no matching model_{}.pt under {}".format(
                    specification["label"],
                    specification["iteration"],
                    log_root,
                ),
                flush=True,
            )
            continue
        candidates.append((specification["label"], checkpoint))

    if args.checkpoint:
        for index, value in enumerate(args.checkpoint):
            checkpoint = Path(value).expanduser().resolve()
            if not checkpoint.is_file():
                raise ValueError(
                    "Extra checkpoint does not exist: " + str(checkpoint)
                )
            candidates.append(
                ("extra_{}_{}".format(index + 1, checkpoint.stem), checkpoint)
            )
    if not candidates:
        raise ValueError(
            "No curated Isaac checkpoints found under " + str(log_root)
        )

    results = []
    for label, checkpoint in candidates:
        output = output_dir / (label + ".csv")
        command = [
            sys.executable,
            str(evaluator),
            "--config_file",
            "n2_stairs_walk.yaml",
            "--checkpoint_path",
            str(checkpoint),
            "--physics_preset",
            "isaac_aligned",
            "--step_height",
            "0.10",
            "--stair_start_x",
            "0.60",
            "--command_speed",
            "0.18",
            "--episodes",
            str(args.episodes),
            "--duration",
            str(args.duration),
            "--output",
            str(output),
            "--seed",
            str(args.seed),
        ]
        print(
            "\n=== {}: {} ===".format(label, checkpoint),
            flush=True,
        )
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        if completed.returncode:
            print(
                "SKIP {}: evaluator exited {}".format(
                    label, completed.returncode
                ),
                flush=True,
            )
            continue
        report_path = output.with_suffix(".json")
        with report_path.open() as report_file:
            summary = json.load(report_file)["summary"]
        summary["candidate"] = label
        summary["checkpoint_path"] = str(checkpoint)
        results.append(summary)

    if not results:
        raise RuntimeError("Every checkpoint evaluation failed")
    ranked = sorted(results, key=_ranking_key)
    print("\n=== MuJoCo transfer ranking ===")
    for rank, result in enumerate(ranked, 1):
        print(
            "{rank}. {candidate}: completion={completion_rate:.1%} "
            "path={path_failure_rate:.1%} fall={fall_rate:.1%} "
            "survival={mean_survival_time_s:.2f}s "
            "yaw={mean_max_yaw_deviation_rad:.3f}rad "
            "distance={mean_forward_distance_m:.3f}m".format(
                rank=rank, **result
            )
        )

    best = ranked[0]
    print(
        "BEST_TRANSFER_CANDIDATE={} checkpoint={}".format(
            best["candidate"], best["checkpoint_path"]
        )
    )
    combined_csv = output_dir / "ranking.csv"
    fields = (
        "candidate",
        "checkpoint_path",
        "completion_rate",
        "success_rate",
        "path_failure_rate",
        "fall_rate",
        "mean_survival_time_s",
        "mean_forward_distance_m",
        "mean_max_lateral_deviation_m",
        "mean_max_yaw_deviation_rad",
        "mean_alternating_tread_rate",
        "mean_same_tread_join_rate",
    )
    with combined_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for result in ranked:
            writer.writerow({field: result[field] for field in fields})
    with (output_dir / "ranking.json").open("w") as output_file:
        json.dump(ranked, output_file, indent=2)
    print("Saved comparison: " + str(combined_csv))
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--output_dir", default=str(_default_output_dir()))
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Optional extra raw model_*.pt; may be specified repeatedly.",
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.episodes < 1:
        raise ValueError("--episodes must be positive")
    if arguments.duration <= 0.0:
        raise ValueError("--duration must be positive")
    compare(arguments)
