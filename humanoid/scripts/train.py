import math
import os
import signal
import sys

# Running ``python humanoid/scripts/train.py`` normally puts this scripts
# directory, not the checkout root, first on sys.path.  Pin the checkout root
# so a sibling/installed ``humanoid`` package cannot be mixed with this file.
_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
if not sys.path or os.path.realpath(sys.path[0]) != _REPOSITORY_ROOT:
    sys.path.insert(0, _REPOSITORY_ROOT)

# Isaac Gym Preview 4 must load its bindings before importing torch.
import isaacgym  # noqa: F401

# 导入所有环境相关模块
from humanoid.envs import *
import torch
# Use the repository-specific name.  Isaac Gym exposes a different
# zero-argument ``get_args`` in some Python 3.8 import orders.
from humanoid.utils.helpers import parse_humanoid_args
from humanoid.utils.task_registry import task_registry


def parse_reward_scale_overrides(value):
    """Parse a comma-separated ``reward=value`` override list."""
    if value is None or not value.strip():
        return {}
    overrides = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "Reward override must use name=value syntax: " + item
            )
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Reward override name cannot be empty")
        try:
            scale = float(raw_value)
        except ValueError as error:
            raise ValueError(
                "Reward override '{}' is not numeric".format(item)
            ) from error
        if not math.isfinite(scale):
            raise ValueError(
                "Reward override '{}' must be finite".format(item)
            )
        overrides[name] = scale
    return overrides


def parse_terrain_level_mix(value):
    """Parse a weighted, comma-separated stair-level mixture."""
    if value is None or not value.strip():
        return None
    levels = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            level = int(item)
        except ValueError as error:
            raise ValueError(
                "Terrain-level mix entries must be integers: " + item
            ) from error
        levels.append(level)
    if not levels:
        raise ValueError("--terrain_level_mix cannot be empty")
    return levels


def train(args):
    """
    训练函数：根据提供的参数执行强化学习训练
    
    参数:
        args: 命令行参数对象，包含训练所需的各种配置
    """
    resume_only_options = (
        args.load_run is not None
        or args.checkpoint is not None
        or args.reset_optimizer
        or args.actor_head_only
        or args.freeze_action_noise
        or args.actor_reference_loss_coeff > 0.0
        or args.symmetrize_actor_reference
        or args.symmetry_loss_coeff > 0.0
        or args.actor_trainable_layers is not None
        or args.reward_scale_overrides is not None
        or args.observation_noise_level is not None
    )
    if resume_only_options and not args.resume:
        raise ValueError(
            "Checkpoint and protected fine-tuning options require --resume"
        )
    if args.actor_head_only and not args.reset_optimizer:
        raise ValueError(
            "--actor_head_only requires --reset_optimizer so stale Adam "
            "moments cannot alter the protected policy"
        )
    if (
        args.actor_trainable_layers is not None
        and args.actor_trainable_layers < 1
    ):
        raise ValueError("--actor_trainable_layers must be positive")
    if args.actor_head_only and args.actor_trainable_layers is not None:
        raise ValueError(
            "--actor_head_only and --actor_trainable_layers are mutually "
            "exclusive"
        )
    if args.actor_trainable_layers is not None and not args.reset_optimizer:
        raise ValueError(
            "--actor_trainable_layers requires --reset_optimizer"
        )
    for option_name in (
        "actor_reference_loss_coeff",
        "symmetry_loss_coeff",
    ):
        value = float(getattr(args, option_name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "--{} must be non-negative and finite".format(option_name)
            )
    if (
        args.symmetrize_actor_reference
        and args.actor_reference_loss_coeff <= 0.0
    ):
        raise ValueError(
            "--symmetrize_actor_reference requires a positive "
            "--actor_reference_loss_coeff"
        )
    if args.symmetrize_actor_reference and args.task != "n2_stairs_walk":
        raise ValueError(
            "--symmetrize_actor_reference currently supports "
            "n2_stairs_walk only"
        )

    # 根据任务名称和参数创建环境实例
    # env: 环境对象，用于模拟和交互
    # env_cfg: 环境配置对象，包含环境的具体配置参数
    env_cfg = None
    stair_tasks = (
        "n2_stairs",
        "n2_stairs_robust",
        "n2_stairs_walk",
    )
    reward_scale_overrides = parse_reward_scale_overrides(
        args.reward_scale_overrides
    )
    terrain_level_mix = parse_terrain_level_mix(args.terrain_level_mix)
    if (
        args.fixed_terrain_level is not None
        and terrain_level_mix is not None
    ):
        raise ValueError(
            "--fixed_terrain_level and --terrain_level_mix are mutually "
            "exclusive"
        )
    if (
        args.fixed_terrain_level is not None
        or terrain_level_mix is not None
        or args.command_speed is not None
        or reward_scale_overrides
        or args.observation_noise_level is not None
    ):
        if args.task not in stair_tasks:
            raise ValueError(
                "Stair terrain, command, reward, and noise overrides are "
                "only valid for n2_stairs tasks"
            )
        env_cfg, _ = task_registry.get_cfgs(name=args.task)
    if args.fixed_terrain_level is not None:
        if not 0 <= args.fixed_terrain_level < env_cfg.terrain.num_rows:
            raise ValueError(
                "--fixed_terrain_level must be in [0, {}]".format(
                    env_cfg.terrain.num_rows - 1
                )
            )
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.fixed_level = args.fixed_terrain_level
        fixed_speed_max = (
            env_cfg.commands.initial_max_speed
            + env_cfg.commands.speed_per_terrain_level
            * args.fixed_terrain_level
        )
        env_cfg.commands.ranges.lin_vel_x[1] = min(
            env_cfg.commands.max_curriculum, fixed_speed_max
        )
        env_cfg.commands.curriculum = False
    if terrain_level_mix is not None:
        invalid_levels = [
            level for level in terrain_level_mix
            if not 0 <= level < env_cfg.terrain.num_rows
        ]
        if invalid_levels:
            raise ValueError(
                "--terrain_level_mix entries must be in [0, {}], received "
                "{}".format(
                    env_cfg.terrain.num_rows - 1,
                    ",".join(str(level) for level in invalid_levels),
                )
            )
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.fixed_level = -1
        env_cfg.terrain.level_mix = list(terrain_level_mix)
        print(
            "Balanced terrain-level mix: {}".format(
                ",".join(str(level) for level in terrain_level_mix)
            )
        )
    if args.command_speed is not None:
        command_speed = float(args.command_speed)
        command_min = float(env_cfg.commands.ranges.lin_vel_x[0])
        command_max = float(env_cfg.commands.max_curriculum)
        if not math.isfinite(command_speed) or not (
            command_min <= command_speed <= command_max
        ):
            raise ValueError(
                "--command_speed must be in [{:.3f}, {:.3f}]".format(
                    command_min, command_max
                )
            )
        env_cfg.commands.ranges.lin_vel_x = [command_speed, command_speed]
        env_cfg.commands.curriculum = False
    for reward_name, reward_scale in reward_scale_overrides.items():
        if not hasattr(env_cfg.rewards.scales, reward_name):
            raise ValueError(
                "Unknown reward scale override: {}".format(reward_name)
            )
        setattr(env_cfg.rewards.scales, reward_name, reward_scale)
        print(
            "Reward-scale override: {}={:.6g}".format(
                reward_name, reward_scale
            )
        )
    if args.observation_noise_level is not None:
        noise_level = float(args.observation_noise_level)
        if not math.isfinite(noise_level) or noise_level < 0.0:
            raise ValueError(
                "--observation_noise_level must be non-negative and finite"
            )
        env_cfg.noise.noise_level = noise_level
        print(
            "Observation-noise level override: {:.3f}".format(noise_level)
        )
    env, env_cfg = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    
    # 创建算法运行器实例
    # ppo_runner: PPO算法运行器对象，负责执行训练过程
    # train_cfg: 训练配置对象，包含训练算法的具体参数
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        load_optimizer=not args.reset_optimizer,
    )
    if args.resume and ppo_runner.current_learning_iteration <= 0:
        raise RuntimeError(
            "Resume requested but checkpoint iteration is not positive; "
            "refusing to silently train from zero."
        )
    if args.learning_rate is not None:
        learning_rate = float(args.learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("--learning_rate must be positive and finite")
        ppo_runner.alg.learning_rate = learning_rate
        ppo_runner.alg_cfg["learning_rate"] = learning_rate
        for param_group in ppo_runner.alg.optimizer.param_groups:
            param_group["lr"] = learning_rate
        print("Learning-rate override: {:.3e}".format(learning_rate))
    if args.fixed_learning_rate:
        ppo_runner.alg.schedule = "fixed"
        ppo_runner.alg_cfg["schedule"] = "fixed"
        print("Learning-rate schedule: fixed")
    if args.action_noise_std is not None:
        action_noise_std = float(args.action_noise_std)
        if not math.isfinite(action_noise_std) or action_noise_std <= 0.0:
            raise ValueError("--action_noise_std must be positive and finite")
        policy = ppo_runner.alg.policy
        with torch.no_grad():
            if policy.noise_std_type == "scalar":
                policy.std.fill_(action_noise_std)
            elif policy.noise_std_type == "log":
                policy.log_std.fill_(math.log(action_noise_std))
            else:
                raise ValueError(
                    "Unsupported policy noise type: "
                    + str(policy.noise_std_type)
                )
        print("Action-noise std override: {:.3f}".format(action_noise_std))
    if args.save_interval is not None:
        save_interval = int(args.save_interval)
        if save_interval < 1:
            raise ValueError("--save_interval must be positive")
        ppo_runner.save_interval = save_interval
        ppo_runner.cfg["runner"]["save_interval"] = save_interval
        train_cfg.runner.save_interval = save_interval
        print("Checkpoint save interval: {}".format(save_interval))
    if args.actor_reference_loss_coeff > 0.0:
        coefficient = float(args.actor_reference_loss_coeff)
        reference_symmetry_env = (
            env if args.symmetrize_actor_reference else None
        )
        ppo_runner.alg.set_actor_reference(
            coefficient, symmetry_env=reference_symmetry_env
        )
        ppo_runner.alg_cfg["actor_reference_loss_coeff"] = coefficient
        ppo_runner.alg_cfg["symmetrize_actor_reference"] = bool(
            args.symmetrize_actor_reference
        )
        print(
            "Actor reference anchor: coefficient={:.4f} teacher={}".format(
                coefficient,
                (
                    "left/right averaged"
                    if args.symmetrize_actor_reference
                    else "raw checkpoint"
                ),
            )
        )
    trainable_actor_layers = args.actor_trainable_layers
    if args.actor_head_only:
        trainable_actor_layers = 1
    if trainable_actor_layers is not None:
        actor = ppo_runner.alg.policy.actor
        linear_layers = [
            module for module in actor.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        if not linear_layers:
            raise RuntimeError("Actor has no Linear output layer to fine-tune")
        if trainable_actor_layers > len(linear_layers):
            raise ValueError(
                "--actor_trainable_layers={} exceeds the Actor's {} Linear "
                "layers".format(trainable_actor_layers, len(linear_layers))
            )
        actor.requires_grad_(False)
        for layer in linear_layers[-trainable_actor_layers:]:
            layer.requires_grad_(True)
        trainable = sum(
            parameter.numel() for parameter in actor.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in actor.parameters())
        print(
            "Actor fine-tuning scope: last {} Linear layer(s) "
            "({}/{} parameters trainable)".format(
                trainable_actor_layers, trainable, total
            )
        )
    if args.freeze_action_noise:
        policy = ppo_runner.alg.policy
        noise_parameter = (
            policy.std
            if policy.noise_std_type == "scalar"
            else policy.log_std
        )
        noise_parameter.requires_grad_(False)
        print("Action-noise parameter: frozen")
    if args.symmetry_loss_coeff > 0.0:
        if args.task != "n2_stairs_walk":
            raise ValueError(
                "--symmetry_loss_coeff currently supports n2_stairs_walk only"
            )
        coefficient = float(args.symmetry_loss_coeff)
        symmetry_cfg = {
            "_env": env,
            "actor_loss_coeff": coefficient,
            "critic_loss_coeff": 0.0,
        }
        ppo_runner.alg.set_symmetry_config(symmetry_cfg)
        ppo_runner.alg_cfg["symmetry_cfg"] = symmetry_cfg
        print(
            "Actor reflection loss: coefficient={:.4f}".format(coefficient)
        )
    
    # max_iterations is treated as the total target iteration. On resume, run
    # only the remainder instead of adding another full training schedule.
    remaining_iterations = max(
        0,
        train_cfg.runner.max_iterations - ppo_runner.current_learning_iteration,
    )
    print(
        "Training iterations: current={}, target={}, remaining={}".format(
            ppo_runner.current_learning_iteration,
            train_cfg.runner.max_iterations,
            remaining_iterations,
        )
    )
    def stop_training(signum, _frame):
        print("Received signal {}; saving a final checkpoint...".format(signum))
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_training)
    try:
        ppo_runner.learn(
            num_learning_iterations=remaining_iterations,
            init_at_random_ep_len=(
                train_cfg.runner.init_at_random_ep_len
                and ppo_runner.current_learning_iteration == 0
            ),
        )
    except KeyboardInterrupt:
        if ppo_runner.log_dir is not None:
            os.makedirs(ppo_runner.log_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                ppo_runner.log_dir,
                "model_{}.pt".format(ppo_runner.current_learning_iteration),
            )
            ppo_runner.save(checkpoint_path)
            print("Saved interrupted training checkpoint: {}".format(checkpoint_path))
        print("Training stopped cleanly.")

# 程序入口点
if __name__ == '__main__':
    # 解析命令行参数
    args = parse_humanoid_args(
        [
            {
                "name": "--fixed_terrain_level",
                "type": int,
                "default": None,
                "help": "Optional fixed n2_stairs row (0=2 cm, ..., 4=10 cm).",
            },
            {
                "name": "--terrain_level_mix",
                "type": str,
                "default": None,
                "help": (
                    "Weighted stair-row mixture, for example "
                    "0,1,2,3,4,4,4,4 keeps half of the environments at "
                    "10 cm."
                ),
            },
            {
                "name": "--reset_optimizer",
                "action": "store_true",
                "default": False,
                "help": (
                    "Load policy/curriculum from a checkpoint but start with "
                    "a fresh optimizer (useful after reward changes)."
                ),
            },
            {
                "name": "--command_speed",
                "type": float,
                "default": None,
                "help": "Fix the forward command for stair specialization.",
            },
            {
                "name": "--learning_rate",
                "type": float,
                "default": None,
                "help": "Override PPO learning rate after checkpoint load.",
            },
            {
                "name": "--fixed_learning_rate",
                "action": "store_true",
                "default": False,
                "help": (
                    "Disable PPO's adaptive KL learning-rate changes for "
                    "controlled fine-tuning."
                ),
            },
            {
                "name": "--action_noise_std",
                "type": float,
                "default": None,
                "help": "Override policy exploration std after checkpoint load.",
            },
            {
                "name": "--save_interval",
                "type": int,
                "default": None,
                "help": "Override checkpoint interval in PPO iterations.",
            },
            {
                "name": "--actor_reference_loss_coeff",
                "type": float,
                "default": 0.0,
                "help": (
                    "Penalize deterministic Actor drift from the checkpoint "
                    "loaded at startup."
                ),
            },
            {
                "name": "--symmetrize_actor_reference",
                "action": "store_true",
                "default": False,
                "help": (
                    "Average the frozen checkpoint teacher with its mirrored "
                    "action, removing checkpoint left/right bias."
                ),
            },
            {
                "name": "--actor_head_only",
                "action": "store_true",
                "default": False,
                "help": "Freeze the Actor trunk and train only its output layer.",
            },
            {
                "name": "--actor_trainable_layers",
                "type": int,
                "default": None,
                "help": (
                    "Freeze the Actor except for its last N Linear layers."
                ),
            },
            {
                "name": "--freeze_action_noise",
                "action": "store_true",
                "default": False,
                "help": "Keep the overridden action-noise parameter fixed.",
            },
            {
                "name": "--symmetry_loss_coeff",
                "type": float,
                "default": 0.0,
                "help": "Exact N2 left/right Actor consistency coefficient.",
            },
            {
                "name": "--reward_scale_overrides",
                "type": str,
                "default": None,
                "help": (
                    "Comma-separated stair reward overrides, for example "
                    "action_rate=-0.3,stairs_lateral_drift=-18."
                ),
            },
            {
                "name": "--observation_noise_level",
                "type": float,
                "default": None,
                "help": "Override the environment observation-noise multiplier.",
            },
        ]
    )
    # 调用训练函数开始训练
    train(args)
