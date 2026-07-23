"""Native MuJoCo PPO fine-tuning for the fixed 10 cm N2 staircase."""

import argparse
import copy
import glob
import json
import os
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
        "logs/n2_stairs_walk/*natural_l4_tiered_8600_s42/model_8600.pt",
        "logs/n2_stairs_walk/*natural_l4_polish_9000_s42/model_9000.pt",
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


def train_configuration(args):
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
            "entropy_coef": 0.002,
            "learning_rate": float(args.learning_rate),
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "schedule": "adaptive",
            "normalize_advantage_per_mini_batch": False,
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
        "Initialized MuJoCo Actor from Isaac iter={} (critic and Adam reset): "
        "{}".format(metadata["iteration"], checkpoint_path),
        flush=True,
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
        "0.10",
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
    return (
        5.0 * float(summary["completion_rate"])
        + 2.0 * float(summary["success_rate"])
        - 2.0 * float(summary["path_failure_rate"])
        - 2.0 * float(summary["fall_rate"])
        + 0.75 * float(summary["mean_alternating_tread_rate"])
        - 0.75 * float(summary["mean_same_tread_join_rate"])
        + 0.25 * float(summary["mean_arm_swing_match"])
        - 0.50
        * float(summary["mean_foot_riser_collision_fraction"])
        + 0.25
        * min(
            float(summary["mean_forward_distance_m"]) / 2.60,
            1.0,
        )
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
        "path={:.1%} fall={:.1%} alternate={:.1%} join={:.1%}".format(
            summary["completion_rate"],
            summary["success_rate"],
            summary["path_failure_rate"],
            summary["fall_rate"],
            summary["mean_alternating_tread_rate"],
            summary["mean_same_tread_join_rate"],
        ),
        flush=True,
    )
    return summary


def train_stage(
    runner,
    env,
    args,
    log_dir,
    target_iteration,
    trunk_trainable,
    best,
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
        if args.skip_eval or args.selection_interval <= 0:
            continue

        checkpoint = log_dir / "model_{}.pt".format(current)
        output = log_dir / "selection_{:05d}.csv".format(current)
        summary = run_mujoco_evaluation(
            args,
            checkpoint,
            output,
            args.selection_episodes,
            args.seed + 20000 + current,
        )
        if summary is None:
            continue
        score = selection_score(summary)
        print(
            "NATIVE_MUJOCO_SELECTION iter={} score={:.4f} "
            "completion={:.1%} success={:.1%} path={:.1%} "
            "fall={:.1%} alternate={:.1%} join={:.1%}".format(
                current,
                score,
                summary["completion_rate"],
                summary["success_rate"],
                summary["path_failure_rate"],
                summary["fall_rate"],
                summary["mean_alternating_tread_rate"],
                summary["mean_same_tread_join_rate"],
            ),
            flush=True,
        )
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
        train_configuration(args),
        log_dir=str(log_dir),
        device=device,
    )
    source_checkpoint = None
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        runner.load(str(resume_path), load_optimizer=True)
        print(
            "Resumed native MuJoCo checkpoint at iteration {}: {}".format(
                runner.current_learning_iteration, resume_path
            ),
            flush=True,
        )
    elif not args.no_warm_start:
        source_checkpoint = (
            auto_init_checkpoint()
            if args.init_checkpoint == "auto"
            else Path(args.init_checkpoint).expanduser().resolve()
        )
        initialize_actor(
            runner, source_checkpoint, args.action_noise_std
        )

    run_metadata = {
        "engine": "mujoco",
        "config": str(config_path),
        "step_height_m": 0.10,
        "num_envs": args.num_envs,
        "num_workers": args.num_workers,
        "device": device,
        "max_iterations": args.max_iterations,
        "freeze_actor_iterations": args.freeze_actor_iterations,
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
    selected_checkpoint = final_checkpoint
    best_checkpoint = log_dir / "model_best.pt"
    if best_checkpoint.is_file():
        selected_checkpoint = best_checkpoint
        print(
            "Selected best native checkpoint from iteration {} "
            "(score={:.4f})".format(
                best["iteration"], best["score"]
            ),
            flush=True,
        )
    if selected_checkpoint == final_checkpoint:
        export_actor(runner, log_dir / "policy.pt")
    else:
        export_checkpoint_actor(
            selected_checkpoint, log_dir / "policy.pt"
        )
    run_acceptance(args, selected_checkpoint, log_dir)
    print(
        "NATIVE_MUJOCO_CHECKPOINT=" + str(selected_checkpoint),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default="n2_stairs_walk.yaml"
    )
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--rollout_steps", type=int, default=48)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument(
        "--freeze_actor_iterations", type=int, default=100
    )
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--action_noise_std", type=float, default=0.25)
    parser.add_argument("--init_checkpoint", default="auto")
    parser.add_argument("--no_warm_start", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run_name", default="native_l4_v1_s42")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument(
        "--device",
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--eval_episodes", type=int, default=16)
    parser.add_argument("--selection_interval", type=int, default=100)
    parser.add_argument("--selection_episodes", type=int, default=4)
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
    if arguments.selection_interval < 0:
        raise ValueError("--selection_interval must be non-negative")
    if arguments.selection_episodes < 1:
        raise ValueError("--selection_episodes must be positive")
    raise SystemExit(main(arguments))
