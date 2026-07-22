import os
import signal

# 导入所有环境相关模块
from humanoid.envs import *
# 导入参数解析和任务注册工具
from humanoid.utils import get_args, task_registry

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
    if args.fixed_terrain_level is not None:
        if args.task not in (
            "n2_stairs",
            "n2_stairs_robust",
            "n2_stairs_walk",
        ):
            raise ValueError("--fixed_terrain_level is only valid for n2_stairs tasks")
        env_cfg, _ = task_registry.get_cfgs(name=args.task)
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
        ]
    )
    # 调用训练函数开始训练
    train(args)
