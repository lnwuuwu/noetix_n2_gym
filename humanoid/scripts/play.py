"""Visualize a checkpoint without silently changing its task definition."""

import os

import isaacgym  # noqa: F401 - Isaac Gym must load before torch
import numpy as np
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *  # noqa: F401,F403 - task registration side effects
from humanoid.utils import export_policy_as_jit, export_policy_as_onnx, get_args, task_registry


def _disable_randomization(env_cfg):
    env_cfg.noise.add_noise = False
    env_cfg.env.test = True
    for name in (
        "randomize_gains",
        "randomize_motor_strength",
        "randomize_base_mass",
        "randomize_com_displacement",
        "randomize_friction",
        "randomize_restitution",
        "push_robots",
        "disturbance",
        "action_delay",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.num_envs is None:
        args.num_envs = 1
    _disable_randomization(env_cfg)

    stair_tasks = ("n2_stairs", "n2_stairs_robust", "n2_stairs_walk")
    if args.task in stair_tasks:
        if not 0 <= args.terrain_level < env_cfg.terrain.num_rows:
            raise ValueError(
                "--terrain_level must be in [0, {}]".format(
                    env_cfg.terrain.num_rows - 1
                )
            )
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.fixed_level = args.terrain_level

    command_min, command_max = env_cfg.commands.ranges.lin_vel_x
    command_speed = args.command_speed
    if command_speed is None:
        command_speed = 0.5 * (command_min + command_max)
    allowed_max = command_max
    if args.task in stair_tasks:
        allowed_max = min(
            env_cfg.commands.max_curriculum,
            env_cfg.commands.initial_max_speed
            + env_cfg.commands.speed_per_terrain_level * args.terrain_level,
        )
    if not command_min <= command_speed <= allowed_max:
        raise ValueError(
            "--command_speed {:.3f} is outside training range [{:.3f}, {:.3f}]".format(
                command_speed, command_min, allowed_max
            )
        )
    env_cfg.commands.resampling_time = [1000, 1001]
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [command_speed, command_speed]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
        load_optimizer=False,
    )
    policy = runner.get_inference_policy(device=env.device)
    # OnPolicyRunner resets the environment during construction, replacing the
    # frame-stacked observation tensor. Fetch the post-reset tensor here.
    obs = env.get_observations()

    if args.export_policy:
        export_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(runner.alg.policy, export_path, runner.obs_normalizer)
        export_policy_as_onnx(runner.alg.policy, export_path, runner.obs_normalizer)
        print("Exported JIT and ONNX policies to: {}".format(export_path))

    camera_offset = np.zeros(3, dtype=np.float64)
    if args.task in stair_tasks:
        camera_offset = env.env_origins[0].detach().cpu().numpy().astype(np.float64)
    camera_position = (
        np.array(env_cfg.viewer.pos, dtype=np.float64) + camera_offset
    )
    camera_target = (
        np.array(env_cfg.viewer.lookat, dtype=np.float64) + camera_offset
    )
    camera_direction = camera_target - camera_position
    if env.viewer is not None:
        env.set_camera(camera_position, camera_target)
    frame_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
        "frames",
    )
    if args.record_frames:
        os.makedirs(frame_dir, exist_ok=True)

    total_steps = args.play_steps
    if total_steps <= 0:
        total_steps = 10 * int(env.max_episode_length)

    completed_episodes = 0
    for step in range(total_steps):
        env.commands[:, 0] = command_speed
        env.commands[:, 1:3] = 0.0
        with torch.inference_mode():
            actions = policy(obs.detach())
        obs, _, _, _, infos, termination_ids, _ = env.step(actions.detach())

        if len(termination_ids) > 0:
            completed_episodes += len(termination_ids)
            if infos.get("episode"):
                compact = {
                    key: float(value.item()) if isinstance(value, torch.Tensor) else value
                    for key, value in infos["episode"].items()
                    if key.startswith("stairs_") or key == "terrain_level"
                }
                if compact:
                    print("episode {}: {}".format(completed_episodes, compact))

        if args.record_frames and step % 2 == 0 and env.viewer is not None:
            env.gym.write_viewer_image_to_file(
                env.viewer, os.path.join(frame_dir, "{:06d}.png".format(step // 2))
            )

        if args.move_camera and env.viewer is not None:
            camera_position[0] += env.dt * 0.25
            env.set_camera(camera_position, camera_position + camera_direction)


if __name__ == "__main__":
    extra_parameters = [
        {
            "name": "--command_speed",
            "type": float,
            "default": None,
            "help": "Forward command in m/s; defaults to the training-range midpoint.",
        },
        {
            "name": "--terrain_level",
            "type": int,
            "default": 0,
            "help": "n2_stairs difficulty row (0=2 cm, ..., 4=10 cm).",
        },
        {
            "name": "--play_steps",
            "type": int,
            "default": 0,
            "help": "Number of policy steps; 0 uses ten episode lengths.",
        },
        {
            "name": "--export_policy",
            "action": "store_true",
            "default": False,
            "help": "Export the loaded Actor as JIT and ONNX.",
        },
        {
            "name": "--record_frames",
            "action": "store_true",
            "default": False,
            "help": "Write viewer frames under logs/<experiment>/exported/frames.",
        },
        {
            "name": "--move_camera",
            "action": "store_true",
            "default": False,
            "help": "Move the viewer camera slowly along +X.",
        },
    ]
    play(get_args(extra_parameters))
