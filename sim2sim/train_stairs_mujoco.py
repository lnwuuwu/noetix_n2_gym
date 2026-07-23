"""Native MuJoCo PPO with deterministic physical-height stair curriculum."""

import argparse
import copy
import glob
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from humanoid.algo.ppo.on_policy_runner import OnPolicyRunner

from eval_stairs_mujoco import load_checkpoint_actor
from mujoco_stairs_env import MujocoStairsVecEnv


def request_safe_stop(signum, _frame):
    """Turn SIGTERM from the launcher into the normal checkpoint path."""
    print(
        "Received signal {}; saving an interrupted checkpoint...".format(
            signum
        ),
        flush=True,
    )
    raise KeyboardInterrupt


class ActorExporter(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()

    def forward(self, observation):
        return self.actor(observation)


def load_config(config_name):
    config_path = Path(config_name)
    if not config_path.is_absolute():
        config_path = ROOT / "sim2sim" / "configs" / config_name
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    def resolve(value):
        return str(
            Path(
                str(value).replace(
                    "{LEGGED_GYM_ROOT_DIR}", str(ROOT)
                )
            )
            .expanduser()
            .resolve()
        )

    config["_resolved_xml_path"] = resolve(config["xml_path"])
    config["_resolved_urdf_path"] = resolve(config["urdf_path"])
    return config, config_path


def auto_init_checkpoint():
    reports = (
        Path("/root/autodl-tmp/n2_eval/isaac_mujoco_compare/ranking.json"),
        ROOT / "reports" / "isaac_mujoco_compare" / "ranking.json",
    )
    for report in reports:
        try:
            with report.open() as report_file:
                ranking = json.load(report_file)
        except (OSError, ValueError, TypeError):
            continue
        by_name = {
            result.get("candidate"): result for result in ranking
        }
        # The comparison proves that every PhysX policy has the same transfer
        # failure.  Prefer the strongest *Isaac* 10 cm natural-gait policy,
        # rather than treating a few tenths of a second before MuJoCo path
        # failure as a meaningful ranking.
        preferred = (
            "natural_l4_9000",
            "tiered_8600",
            "natural_fast_8000",
            "phase_v2_5000",
        )
        ordered = [
            by_name[name] for name in preferred if name in by_name
        ]
        ordered.extend(
            result for result in ranking if result not in ordered
        )
        for result in ordered:
            checkpoint = Path(result["checkpoint_path"])
            if checkpoint.is_file():
                return checkpoint

    patterns = (
        "logs/n2_stairs_walk/*natural_l4_polish_9000_s42/model_9000.pt",
        "logs/n2_stairs_walk/*natural_l4_tiered_8600_s42/model_8600.pt",
    )
    for pattern in patterns:
        matches = [
            Path(value)
            for value in glob.glob(str(ROOT / pattern))
        ]
        if matches:
            return max(
                matches, key=lambda path: path.stat().st_mtime
            )
    raise ValueError(
        "No 410-D Isaac initialization checkpoint found. Pass "
        "--init_checkpoint=/absolute/path/model_*.pt"
    )


def auto_native_checkpoint():
    candidates = [
        Path(value)
        for value in glob.glob(
            str(
                ROOT
                / "logs_mujoco"
                / "n2_stairs_walk"
                / "*"
                / "model_best.pt"
            )
        )
        if "smoke" not in Path(value).parent.name
    ]
    if not candidates:
        raise ValueError(
            "No native MuJoCo model_best.pt exists; use "
            "--init_checkpoint=auto for the Isaac initialization"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def auto_v2_actor_checkpoint():
    """Reuse only the best v2 Actor; reset its critic, Adam, and curriculum."""
    candidates = [
        Path(value)
        for value in glob.glob(
            str(
                ROOT
                / "logs_mujoco"
                / "n2_stairs_walk"
                / "*mujoco_curriculum_v2_*"
                / "model_best.pt"
            )
        )
        if "smoke" not in Path(value).parent.name
    ]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    print(
        "No v2 pilot Actor found; falling back to the Isaac Actor.",
        flush=True,
    )
    return auto_init_checkpoint()


def train_configuration(args, env):
    return {
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": float(args.action_noise_std),
            "noise_std_type": "scalar",
        },
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.001,
            "learning_rate": float(args.learning_rate),
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "gamma": 0.997,
            "lam": 0.95,
            "desired_kl": 0.008,
            "schedule": "adaptive",
            "normalize_advantage_per_mini_batch": False,
            "symmetry_cfg": {
                "_env": env,
                "actor_loss_coeff": float(
                    args.symmetry_loss_coeff
                ),
                "critic_loss_coeff": float(
                    args.critic_symmetry_loss_coeff
                ),
            },
        },
        "runner": {
            "num_steps_per_env": int(args.rollout_steps),
            "save_interval": int(args.save_interval),
            "empirical_normalization": False,
        },
        "logger": "tensorboard",
    }


def set_actor_trunk_trainable(policy, trainable):
    linears = [
        module
        for module in policy.actor
        if isinstance(module, torch.nn.Linear)
    ]
    if len(linears) < 2:
        raise RuntimeError("Actor has no feature trunk to freeze")
    for linear in linears[:-1]:
        for parameter in linear.parameters():
            parameter.requires_grad_(bool(trainable))
    # The output head always remains trainable.
    for parameter in linears[-1].parameters():
        parameter.requires_grad_(True)


def initialize_actor(runner, checkpoint_path, action_noise_std):
    source, metadata = load_checkpoint_actor(str(checkpoint_path))
    if metadata["input_dim"] != runner.env.num_obs:
        raise ValueError(
            "Warm-start Actor input {} != MuJoCo observation {}".format(
                metadata["input_dim"], runner.env.num_obs
            )
        )
    if metadata["output_dim"] != runner.env.num_actions:
        raise ValueError(
            "Warm-start Actor output {} != {}".format(
                metadata["output_dim"], runner.env.num_actions
            )
        )
    runner.alg.policy.actor.load_state_dict(source.actor.state_dict())
    with torch.no_grad():
        if hasattr(runner.alg.policy, "std"):
            runner.alg.policy.std.fill_(float(action_noise_std))
        else:
            runner.alg.policy.log_std.fill_(
                float(torch.log(torch.tensor(action_noise_std)))
            )
    print(
        "Initialized MuJoCo Actor iter={} (critic and Adam reset): "
        "{}".format(metadata["iteration"], checkpoint_path),
        flush=True,
    )


def set_policy_noise_std(policy, action_noise_std):
    """Set exploration after resume without changing Actor weights."""
    with torch.no_grad():
        if hasattr(policy, "std"):
            policy.std.fill_(float(action_noise_std))
        else:
            policy.log_std.fill_(
                float(torch.log(torch.tensor(action_noise_std)))
            )


def export_actor(runner, output_path):
    exporter = ActorExporter(runner.alg.policy.actor)
    exporter.eval()
    scripted = torch.jit.script(exporter)
    scripted.save(str(output_path))
    print("Exported MuJoCo policy: " + str(output_path), flush=True)


def export_checkpoint_actor(checkpoint_path, output_path):
    source, metadata = load_checkpoint_actor(str(checkpoint_path))
    exporter = ActorExporter(source.actor)
    exporter.eval()
    torch.jit.script(exporter).save(str(output_path))
    print(
        "Exported selected MuJoCo policy iter={}: {}".format(
            metadata["iteration"], output_path
        ),
        flush=True,
    )


def run_mujoco_evaluation(
    args,
    checkpoint_path,
    output,
    episodes,
    seed,
    step_height=0.10,
):
    command = [
        sys.executable,
        str(ROOT / "sim2sim" / "eval_stairs_mujoco.py"),
        "--config_file",
        str(args.config_file),
        "--checkpoint_path",
        str(checkpoint_path),
        "--physics_preset",
        "isaac_aligned",
        "--step_height",
        str(float(step_height)),
        "--stair_start_x",
        "0.60",
        "--command_speed",
        "0.18",
        "--episodes",
        str(episodes),
        "--output",
        str(output),
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    if completed.returncode:
        print(
            "MuJoCo evaluation failed with exit code {}".format(
                completed.returncode
            ),
            flush=True,
        )
        return None
    with output.with_suffix(".json").open() as report_file:
        summary = json.load(report_file)["summary"]
    return summary


def selection_score(summary):
    """Balance physical completion with strict natural-gait quality."""
    expected_climb = max(
        float(summary["step_height_m"]) * 6.0, 1.0e-6
    )
    return (
        20.0 * float(summary["success_rate"])
        + 8.0 * float(summary["completion_rate"])
        - 3.0 * float(summary["path_failure_rate"])
        - 3.0 * float(summary["fall_rate"])
        + 2.0 * float(summary["mean_alternating_tread_rate"])
        - 1.5 * float(summary["mean_same_tread_join_rate"])
        + 0.25 * float(summary.get("mean_arm_swing_match", 0.0))
        - 1.0
        * float(summary["mean_foot_riser_collision_fraction"])
        - 0.5 * float(summary["mean_max_lateral_deviation_m"])
        - 0.5 * float(summary["mean_max_yaw_deviation_rad"])
        - 2.0
        * min(
            abs(
                float(summary["mean_forward_speed_m_s"])
                - float(summary["command_speed_m_s"])
            )
            / 0.10,
            2.0,
        )
        + 1.0
        * min(
            float(summary["mean_climb_height_m"]) / expected_climb,
            1.0,
        )
        + 0.5
        * min(
            float(summary["mean_forward_distance_m"]) / 2.60,
            1.0,
        )
    )


def physical_promotion_readiness(env, summary):
    """Rank intermediate policies by how many height-gate clauses they meet."""
    gate = env.curriculum_cfg["physical_promotion"]
    expected_climb = (
        int(env.stair_cfg["num_steps"]) * env.physical_step_height
    )
    speed_error = abs(
        float(summary["mean_forward_speed_m_s"])
        - float(summary["command_speed_m_s"])
    )
    checks = (
        math.isclose(
            float(summary["step_height_m"]),
            env.physical_step_height,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ),
        float(summary["completion_rate"])
        >= float(gate["min_completion_rate"]),
        float(summary["fall_rate"]) <= float(gate["max_fall_rate"]),
        float(summary["path_failure_rate"])
        <= float(gate["max_path_failure_rate"]),
        float(summary["mean_climb_height_m"])
        >= env._physical_gate_value(gate, "min_climb_fraction")
        * expected_climb,
        float(summary["mean_alternating_tread_rate"])
        >= env._physical_gate_value(
            gate, "min_alternating_tread_rate"
        ),
        float(summary["mean_same_tread_join_rate"])
        <= env._physical_gate_value(
            gate, "max_same_tread_join_rate"
        ),
        speed_error
        <= env._physical_gate_value(gate, "max_speed_error_m_s"),
        float(summary["mean_max_yaw_deviation_rad"])
        <= env._physical_gate_value(gate, "max_yaw_deviation_rad"),
    )
    # A policy satisfying one more hard gate must outrank any cosmetic score
    # gain. This preserves the speed-controlled model_300-like candidate over
    # a faster model with slightly higher completion but two failed gates.
    return 100.0 * sum(bool(value) for value in checks) + selection_score(
        summary
    )


def checkpoint_gate_passed(summary, curriculum_cfg):
    """Require deterministic 10 cm climbing before naming a model best."""
    heights = curriculum_cfg["physical_step_heights_m"]
    final_height = float(heights[-1])
    gate = curriculum_cfg["checkpoint_gate"]
    expected_climb = final_height * 6.0
    return (
        math.isclose(
            float(summary["step_height_m"]),
            final_height,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        and float(summary["completion_rate"])
        >= float(gate["min_completion_rate"])
        and float(summary["success_rate"])
        >= float(gate["min_success_rate"])
        and float(summary["fall_rate"])
        <= float(gate["max_fall_rate"])
        and float(summary["path_failure_rate"])
        <= float(gate["max_path_failure_rate"])
        and float(summary["mean_climb_height_m"])
        >= float(gate["min_climb_fraction"]) * expected_climb
        and float(summary["mean_alternating_tread_rate"])
        >= float(gate["min_alternating_tread_rate"])
        and float(summary["mean_same_tread_join_rate"])
        <= float(gate["max_same_tread_join_rate"])
        and abs(
            float(summary["mean_forward_speed_m_s"])
            - float(summary["command_speed_m_s"])
        )
        <= float(gate["max_speed_error_m_s"])
        and float(summary["mean_max_lateral_deviation_m"])
        <= float(gate["max_lateral_deviation_m"])
        and float(summary["mean_max_yaw_deviation_rad"])
        <= float(gate["max_yaw_deviation_rad"])
    )


def run_acceptance(args, checkpoint_path, log_dir):
    if args.skip_eval:
        return None
    output = log_dir / "mujoco_acceptance.csv"
    summary = run_mujoco_evaluation(
        args,
        checkpoint_path,
        output,
        args.eval_episodes,
        args.seed + 10000,
    )
    if summary is None:
        return None
    print(
        "NATIVE_MUJOCO_ACCEPTANCE completion={:.1%} success={:.1%} "
        "path={:.1%} fall={:.1%} yaw={:.3f}rad "
        "distance={:.3f}m climb={:.3f}m "
        "alternate={:.1%} join={:.1%}".format(
            summary["completion_rate"],
            summary["success_rate"],
            summary["path_failure_rate"],
            summary["fall_rate"],
            summary["mean_max_yaw_deviation_rad"],
            summary["mean_forward_distance_m"],
            summary["mean_climb_height_m"],
            summary["mean_alternating_tread_rate"],
            summary["mean_same_tread_join_rate"],
        ),
        flush=True,
    )
    return summary


def promotion_evaluation_seed(args, env):
    """Use disjoint deterministic batches for consecutive height gates."""
    if env.physical_curriculum_complete:
        return int(args.seed) + 20000
    return (
        int(args.seed) + 20000
        + int(env.physical_height_index) * 1000
        + int(env.physical_promotion_streak)
        * int(args.selection_episodes)
    )


def restore_progress_best(log_dir):
    """Restore the best deterministic candidate at the furthest height."""
    state = {
        "height_index": -1,
        "height_m": 0.0,
        "score": float("-inf"),
        "readiness_score": float("-inf"),
        "iteration": None,
        "summary": None,
    }
    path = log_dir / "model_progress_best.json"
    if not path.is_file():
        return state
    try:
        with path.open() as state_file:
            saved = json.load(state_file)
        if (
            int(saved["height_index"]) >= 0
            and math.isfinite(float(saved["score"]))
            and (log_dir / "model_progress_best.pt").is_file()
        ):
            state.update(saved)
    except (OSError, ValueError, KeyError, TypeError):
        print(
            "Previous model_progress_best.json is invalid; selecting again.",
            flush=True,
        )
    return state


def update_progress_best(
    log_dir,
    checkpoint,
    iteration,
    height_index,
    summary,
    score,
    readiness_score,
    progress_best,
):
    """Never lose an earlier stable policy to later PPO regression."""
    improved = (
        int(height_index) > int(progress_best["height_index"])
        or (
            int(height_index) == int(progress_best["height_index"])
            and float(readiness_score)
            > float(progress_best["readiness_score"])
        )
    )
    if not improved:
        return False
    progress_best.update(
        height_index=int(height_index),
        height_m=float(summary["step_height_m"]),
        score=float(score),
        readiness_score=float(readiness_score),
        iteration=int(iteration),
        summary=summary,
    )
    shutil.copy2(checkpoint, log_dir / "model_progress_best.pt")
    with (log_dir / "model_progress_best.json").open("w") as state_file:
        json.dump(progress_best, state_file, indent=2)
    print(
        "NATIVE_MUJOCO_PROGRESS_BEST iter={} height={:.2f}m "
        "score={:.4f} readiness={:.1f} completion={:.1%} fall={:.1%} "
        "alternate={:.1%} join={:.1%}".format(
            iteration,
            summary["step_height_m"],
            score,
            readiness_score,
            summary["completion_rate"],
            summary["fall_rate"],
            summary["mean_alternating_tread_rate"],
            summary["mean_same_tread_join_rate"],
        ),
        flush=True,
    )
    return True


def robust_checkpoint_tournament(
    args, log_dir, best, curriculum_cfg
):
    """Re-evaluate top periodic candidates on one larger fixed seed set."""
    if args.skip_eval or args.tournament_candidates <= 0:
        return None
    candidates = []
    for report_path in sorted(log_dir.glob("selection_*.json")):
        try:
            iteration = int(report_path.stem.split("_")[-1])
            with report_path.open() as report_file:
                summary = json.load(report_file)["summary"]
            checkpoint = log_dir / "model_{}.pt".format(iteration)
            if (
                checkpoint.is_file()
                and checkpoint_gate_passed(summary, curriculum_cfg)
            ):
                candidates.append(
                    (
                        selection_score(summary),
                        iteration,
                        checkpoint,
                    )
                )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    candidates.sort(reverse=True)
    candidates = candidates[:int(args.tournament_candidates)]
    if not candidates:
        return None

    winner = None
    for _, iteration, checkpoint in candidates:
        output = log_dir / "tournament_{:05d}.csv".format(iteration)
        summary = run_mujoco_evaluation(
            args,
            checkpoint,
            output,
            args.tournament_episodes,
            args.seed + 30000,
        )
        if summary is None:
            continue
        if not checkpoint_gate_passed(summary, curriculum_cfg):
            print(
                "NATIVE_MUJOCO_TOURNAMENT_REJECT iter={} "
                "completion={:.1%} path={:.1%} fall={:.1%}".format(
                    iteration,
                    summary["completion_rate"],
                    summary["path_failure_rate"],
                    summary["fall_rate"],
                ),
                flush=True,
            )
            continue
        score = selection_score(summary)
        print(
            "NATIVE_MUJOCO_TOURNAMENT iter={} score={:.4f} "
            "completion={:.1%} success={:.1%} path={:.1%} "
            "fall={:.1%} yaw={:.3f}rad distance={:.3f}m "
            "alternate={:.1%} join={:.1%}".format(
                iteration,
                score,
                summary["completion_rate"],
                summary["success_rate"],
                summary["path_failure_rate"],
                summary["fall_rate"],
                summary["mean_max_yaw_deviation_rad"],
                summary["mean_forward_distance_m"],
                summary["mean_alternating_tread_rate"],
                summary["mean_same_tread_join_rate"],
            ),
            flush=True,
        )
        if winner is None or score > winner["score"]:
            winner = {
                "score": score,
                "iteration": iteration,
                "summary": summary,
                "checkpoint": checkpoint,
            }
    if winner is None:
        return None
    shutil.copy2(winner["checkpoint"], log_dir / "model_best.pt")
    best.update(
        score=winner["score"],
        iteration=winner["iteration"],
        summary=winner["summary"],
    )
    with (log_dir / "model_best.json").open("w") as best_file:
        json.dump(best, best_file, indent=2)
    print(
        "NATIVE_MUJOCO_ROBUST_BEST iter={} score={:.4f}".format(
            best["iteration"], float(best["score"])
        ),
        flush=True,
    )
    return winner


def train_stage(
    runner,
    env,
    args,
    log_dir,
    target_iteration,
    trunk_trainable,
    best,
    progress_best,
):
    set_actor_trunk_trainable(runner.alg.policy, trunk_trainable)
    env.set_adaptation_stage(not trunk_trainable)
    while runner.current_learning_iteration < target_iteration:
        current = int(runner.current_learning_iteration)
        chunk = target_iteration - current
        if not args.skip_eval and args.selection_interval > 0:
            chunk = min(chunk, int(args.selection_interval))
        runner.learn(chunk)
        current = int(runner.current_learning_iteration)
        if hasattr(env, "curriculum_summary"):
            curriculum = env.curriculum_summary()
            print(
                "NATIVE_MUJOCO_CURRICULUM iter={} mean={:.2f} "
                "max={} final={:.1%} height={:.2f}m "
                "height_gate={}/{} levels={}".format(
                    current,
                    curriculum["mean_mastery_level"],
                    curriculum["max_mastery_level"],
                    curriculum["final_level_fraction"],
                    curriculum["physical_step_height_m"],
                    curriculum["physical_promotion_streak"],
                    env.curriculum_cfg["physical_promotion"][
                        "consecutive_evaluations"
                    ],
                    curriculum["level_histogram"],
                ),
                flush=True,
            )
        if args.skip_eval or args.selection_interval <= 0:
            continue

        checkpoint = log_dir / "model_{}.pt".format(current)
        output = log_dir / "selection_{:05d}.csv".format(current)
        summary = run_mujoco_evaluation(
            args,
            checkpoint,
            output,
            args.selection_episodes,
            promotion_evaluation_seed(args, env),
            step_height=env.physical_step_height,
        )
        if summary is None:
            continue
        score = selection_score(summary)
        readiness_score = physical_promotion_readiness(env, summary)
        print(
            "NATIVE_MUJOCO_SELECTION iter={} score={:.4f} "
            "height={:.2f}m completion={:.1%} success={:.1%} path={:.1%} "
            "fall={:.1%} speed={:.3f}m/s speederr={:.3f}m/s "
            "yaw={:.3f}rad distance={:.3f}m "
            "climb={:.3f}m alternate={:.1%} join={:.1%}".format(
                current,
                score,
                summary["step_height_m"],
                summary["completion_rate"],
                summary["success_rate"],
                summary["path_failure_rate"],
                summary["fall_rate"],
                summary["mean_forward_speed_m_s"],
                abs(
                    summary["mean_forward_speed_m_s"]
                    - summary["command_speed_m_s"]
                ),
                summary["mean_max_yaw_deviation_rad"],
                summary["mean_forward_distance_m"],
                summary["mean_climb_height_m"],
                summary["mean_alternating_tread_rate"],
                summary["mean_same_tread_join_rate"],
            ),
            flush=True,
        )
        promotion_gate = env.curriculum_cfg["physical_promotion"]
        evaluated_height_index = int(env.physical_height_index)
        height_gate_passed = (
            env.physical_curriculum_complete
            or env._physical_gate_passed(summary, promotion_gate)
        )
        promoted = env.update_physical_curriculum(summary)
        reported_gate_streak = (
            int(promotion_gate["consecutive_evaluations"])
            if promoted
            else int(env.physical_promotion_streak)
        )
        # runner.learn() saved before deterministic evaluation. Persist the
        # updated gate streak/height in the same numeric checkpoint so resume
        # cannot silently fall back one physical stage.
        runner.save(str(checkpoint))
        update_progress_best(
            log_dir,
            checkpoint,
            current,
            evaluated_height_index,
            summary,
            score,
            readiness_score,
            progress_best,
        )
        print(
            "NATIVE_MUJOCO_HEIGHT_GATE iter={} passed={} streak={}/{} "
            "height={:.2f}m alternate={:.1%} join={:.1%} "
            "speederr={:.3f}m/s yaw={:.3f}rad".format(
                current,
                height_gate_passed,
                reported_gate_streak,
                promotion_gate["consecutive_evaluations"],
                summary["step_height_m"],
                summary["mean_alternating_tread_rate"],
                summary["mean_same_tread_join_rate"],
                abs(
                    summary["mean_forward_speed_m_s"]
                    - summary["command_speed_m_s"]
                ),
                summary["mean_max_yaw_deviation_rad"],
            ),
            flush=True,
        )
        if promoted:
            print(
                "NATIVE_MUJOCO_CURRICULUM_RESET iter={} height={:.2f}m".format(
                    current, env.physical_step_height
                ),
                flush=True,
            )
        if not checkpoint_gate_passed(summary, env.curriculum_cfg):
            print(
                "NATIVE_MUJOCO_BEST_REJECT iter={} height={:.2f}m "
                "completion={:.1%} success={:.1%} path={:.1%} "
                "fall={:.1%} climb={:.3f}m speederr={:.3f}m/s "
                "alternate={:.1%} join={:.1%}".format(
                    current,
                    summary["step_height_m"],
                    summary["completion_rate"],
                    summary["success_rate"],
                    summary["path_failure_rate"],
                    summary["fall_rate"],
                    summary["mean_climb_height_m"],
                    abs(
                        summary["mean_forward_speed_m_s"]
                        - summary["command_speed_m_s"]
                    ),
                    summary["mean_alternating_tread_rate"],
                    summary["mean_same_tread_join_rate"],
                ),
                flush=True,
            )
            continue
        if score > best["score"]:
            best.update(
                score=score,
                iteration=current,
                summary=summary,
            )
            shutil.copy2(checkpoint, log_dir / "model_best.pt")
            with (log_dir / "model_best.json").open("w") as best_file:
                json.dump(best, best_file, indent=2)
            print(
                "NATIVE_MUJOCO_NEW_BEST iter={} score={:.4f}".format(
                    current, score
                ),
                flush=True,
            )


def main(args):
    if args.smoke:
        args.num_envs = 4
        args.num_workers = 2
        args.max_iterations = 5
        args.freeze_actor_iterations = 0
        args.rollout_steps = 16
        args.save_interval = 5
        args.skip_eval = True

    torch.manual_seed(args.seed)
    config, config_path = load_config(args.config_file)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU for PPO", flush=True)
        device = "cpu"

    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser().resolve()
    else:
        timestamp = time.strftime("%m%d_%H-%M-%S")
        log_dir = (
            ROOT
            / "logs_mujoco"
            / "n2_stairs_walk"
            / (timestamp + "_" + args.run_name)
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = log_dir / "mujoco_run.json"

    env = MujocoStairsVecEnv(
        config,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    runner = OnPolicyRunner(
        env,
        train_configuration(args, env),
        log_dir=str(log_dir),
        device=device,
    )
    source_checkpoint = None
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        runner.load(str(resume_path), load_optimizer=True)
        if args.resume_action_noise_std is not None:
            set_policy_noise_std(
                runner.alg.policy, args.resume_action_noise_std
            )
        print(
            "Resumed native MuJoCo checkpoint at iteration {}: {}".format(
                runner.current_learning_iteration, resume_path
            ),
            flush=True,
        )
        if args.resume_action_noise_std is not None:
            print(
                "Reset resumed exploration std to {:.3f}".format(
                    args.resume_action_noise_std
                ),
                flush=True,
            )
    elif not args.no_warm_start:
        if args.init_checkpoint == "auto":
            source_checkpoint = auto_init_checkpoint()
        elif args.init_checkpoint == "auto_native":
            source_checkpoint = auto_native_checkpoint()
        elif args.init_checkpoint == "auto_v2":
            source_checkpoint = auto_v2_actor_checkpoint()
        else:
            source_checkpoint = (
                Path(args.init_checkpoint).expanduser().resolve()
            )
        initialize_actor(
            runner, source_checkpoint, args.action_noise_std
        )

    run_metadata = {
        "engine": "mujoco",
        "config": str(config_path),
        "physical_step_heights_m": list(
            map(
                float,
                config["mujoco_training"]["curriculum"][
                    "physical_step_heights_m"
                ],
            )
        ),
        "num_envs": args.num_envs,
        "num_workers": args.num_workers,
        "device": device,
        "max_iterations": args.max_iterations,
        "freeze_actor_iterations": args.freeze_actor_iterations,
        "curriculum_version": 7,
        "symmetry_loss_coeff": args.symmetry_loss_coeff,
        "selection_episodes": args.selection_episodes,
        "tournament_episodes": args.tournament_episodes,
        "source_checkpoint": (
            str(source_checkpoint) if source_checkpoint else None
        ),
        "resume": args.resume,
        "seed": args.seed,
    }
    with metadata_path.open("w") as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2)
    print("MuJoCo run directory: " + str(log_dir), flush=True)
    best = {
        "score": float("-inf"),
        "iteration": None,
        "summary": None,
    }
    progress_best = restore_progress_best(log_dir)
    if progress_best["iteration"] is not None:
        print(
            "Restored progress-best selection: iter={} height={:.2f}m "
            "score={:.4f}".format(
                progress_best["iteration"],
                progress_best["height_m"],
                progress_best["score"],
            ),
            flush=True,
        )
    previous_best_path = log_dir / "model_best.json"
    if args.resume and previous_best_path.is_file():
        try:
            with previous_best_path.open() as previous_best_file:
                previous_best = json.load(previous_best_file)
            if math.isfinite(float(previous_best["score"])):
                best.update(previous_best)
                print(
                    "Restored previous best native selection: "
                    "iter={} score={:.4f}".format(
                        best["iteration"], float(best["score"])
                    ),
                    flush=True,
                )
        except (OSError, ValueError, KeyError, TypeError):
            print(
                "Previous model_best.json is invalid; selecting again.",
                flush=True,
            )

    signal.signal(signal.SIGTERM, request_safe_stop)
    try:
        current = int(runner.current_learning_iteration)
        target = int(args.max_iterations)
        if current > target:
            raise ValueError(
                "Native checkpoint iteration {} exceeds target {}".format(
                    current, target
                )
            )
        adaptation_target = min(
            target, int(args.freeze_actor_iterations)
        )
        if current < adaptation_target:
            train_stage(
                runner,
                env,
                args,
                log_dir,
                adaptation_target,
                False,
                best,
                progress_best,
            )
            current = int(runner.current_learning_iteration)
        if current < target:
            train_stage(
                runner,
                env,
                args,
                log_dir,
                target,
                True,
                best,
                progress_best,
            )
    except KeyboardInterrupt:
        interrupted_path = log_dir / "model_interrupted.pt"
        runner.save(str(interrupted_path))
        print(
            "Interrupted safely; saved " + str(interrupted_path),
            flush=True,
        )
        return 130
    finally:
        env.close()

    final_checkpoint = (
        log_dir
        / "model_{}.pt".format(runner.current_learning_iteration)
    )
    if not final_checkpoint.is_file():
        runner.save(str(final_checkpoint))
    curriculum_cfg = config["mujoco_training"]["curriculum"]
    robust_checkpoint_tournament(
        args, log_dir, best, curriculum_cfg
    )
    best_checkpoint = log_dir / "model_best.pt"
    progress_checkpoint = log_dir / "model_progress_best.pt"
    if args.skip_eval:
        export_actor(runner, log_dir / "policy.pt")
        print(
            "NATIVE_MUJOCO_CHECKPOINT=" + str(final_checkpoint),
            flush=True,
        )
        return 0

    candidate = (
        best_checkpoint
        if best_checkpoint.is_file()
        else (
            progress_checkpoint
            if progress_checkpoint.is_file()
            else final_checkpoint
        )
    )
    acceptance = run_acceptance(args, candidate, log_dir)
    accepted = (
        acceptance is not None
        and checkpoint_gate_passed(acceptance, curriculum_cfg)
    )
    if accepted:
        if candidate != best_checkpoint:
            shutil.copy2(candidate, best_checkpoint)
            best.update(
                score=selection_score(acceptance),
                iteration=int(runner.current_learning_iteration),
                summary=acceptance,
            )
            with (log_dir / "model_best.json").open("w") as best_file:
                json.dump(best, best_file, indent=2)
        export_checkpoint_actor(
            best_checkpoint, log_dir / "policy.pt"
        )
        print(
            "NATIVE_MUJOCO_ACCEPTED checkpoint={}".format(
                best_checkpoint
            ),
            flush=True,
        )
        print(
            "NATIVE_MUJOCO_CHECKPOINT=" + str(best_checkpoint),
            flush=True,
        )
        return 0

    rejected_checkpoint = log_dir / "model_rejected.pt"
    if best_checkpoint.is_file():
        os.replace(best_checkpoint, rejected_checkpoint)
        rejected_source = rejected_checkpoint
    else:
        shutil.copy2(candidate, rejected_checkpoint)
        rejected_source = candidate
    best_json = log_dir / "model_best.json"
    if best_json.is_file():
        os.replace(best_json, log_dir / "model_rejected.json")
    elif (log_dir / "model_progress_best.json").is_file():
        shutil.copy2(
            log_dir / "model_progress_best.json",
            log_dir / "model_rejected.json",
        )
    export_checkpoint_actor(
        rejected_source, log_dir / "policy_rejected.pt"
    )
    print(
        "NATIVE_MUJOCO_REJECTED no checkpoint met the deterministic "
        "10 cm acceptance gate",
        flush=True,
    )
    print(
        "NATIVE_MUJOCO_REJECTED_CHECKPOINT="
        + str(rejected_checkpoint),
        flush=True,
    )
    print("NATIVE_MUJOCO_CHECKPOINT=NONE", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="n2_stairs_walk.yaml"
    )
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=1200)
    parser.add_argument("--rollout_steps", type=int, default=96)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument(
        "--freeze_actor_iterations", type=int, default=0
    )
    parser.add_argument("--learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--action_noise_std", type=float, default=0.20)
    parser.add_argument(
        "--resume_action_noise_std", type=float, default=None
    )
    parser.add_argument(
        "--symmetry_loss_coeff", type=float, default=0.50
    )
    parser.add_argument(
        "--critic_symmetry_loss_coeff", type=float, default=0.05
    )
    parser.add_argument("--init_checkpoint", default="auto")
    parser.add_argument("--no_warm_start", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--run_name", default="mujoco_curriculum_v7_s42"
    )
    parser.add_argument("--log_dir", default=None)
    parser.add_argument(
        "--device",
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--eval_episodes", type=int, default=32)
    parser.add_argument("--selection_interval", type=int, default=100)
    parser.add_argument("--selection_episodes", type=int, default=16)
    parser.add_argument("--tournament_candidates", type=int, default=3)
    parser.add_argument("--tournament_episodes", type=int, default=32)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.num_envs < 1 or arguments.num_workers < 1:
        raise ValueError("Environment and worker counts must be positive")
    if arguments.max_iterations < 0:
        raise ValueError("--max_iterations must be non-negative")
    if arguments.rollout_steps < 1:
        raise ValueError("--rollout_steps must be positive")
    if arguments.action_noise_std <= 0.0:
        raise ValueError("--action_noise_std must be positive")
    if (
        arguments.resume_action_noise_std is not None
        and arguments.resume_action_noise_std <= 0.0
    ):
        raise ValueError("--resume_action_noise_std must be positive")
    if arguments.selection_interval < 0:
        raise ValueError("--selection_interval must be non-negative")
    if arguments.selection_episodes < 1:
        raise ValueError("--selection_episodes must be positive")
    if arguments.tournament_episodes < 1:
        raise ValueError("--tournament_episodes must be positive")
    if arguments.tournament_candidates < 0:
        raise ValueError("--tournament_candidates must be non-negative")
    if (
        arguments.symmetry_loss_coeff < 0.0
        or arguments.critic_symmetry_loss_coeff < 0.0
    ):
        raise ValueError("Symmetry loss coefficients must be non-negative")
    raise SystemExit(main(arguments))
