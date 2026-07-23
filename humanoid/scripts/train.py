import math
import os
import signal

# 导入所有环境相关模块
from humanoid.envs import *
# Isaac Gym Preview 4 must load its bindings before importing torch.
import torch
# Import the parser from its defining module.  Python 3.8 can otherwise bind a
# same-named symbol exposed while the lazy ``humanoid.utils`` package imports
# task-registration side effects, yielding a zero-argument ``get_args`` here.
from humanoid.utils.helpers import get_args
from humanoid.utils import task_registry

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
    )
    if resume_only_options and not args.resume:
        raise ValueError(
            "--load_run, --checkpoint, and --reset_optimizer require --resume"
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
    if (
        args.fixed_terrain_level is not None
        or args.command_speed is not None
    ):
        if args.task not in stair_tasks:
            raise ValueError(
                "--fixed_terrain_level and --command_speed are only valid "
                "for n2_stairs tasks"
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
    args = get_args(
        [
            {
                "name": "--fixed_terrain_level",
                "type": int,
                "default": None,
                "help": "Optional fixed n2_stairs row (0=2 cm, ..., 4=10 cm).",
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
        ]
    )
    # 调用训练函数开始训练
    train(args)
