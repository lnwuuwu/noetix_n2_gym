#!/usr/bin/env python3
"""从已训练好的策略 checkpoint 中采集参考动作数据，供 AMP 判别器使用。

输出 JSON 格式与 MotionLoaderNing 兼容:
  { "Frames": [[62 floats], ...], "MotionWeight": 1.0, "FrameDuration": 0.02 }

62 维全帧 = root_pos(3) + root_rot(4) + joint_pose(18) + toe_pos_local(12)
             + lin_vel(3) + ang_vel(3) + joint_vel(18) + base_height(1)

用法:
  python humanoid/scripts/collect_reference_motions.py \
    --checkpoint path/to/model_best.pt \
    --output humanoid/amp_data/stair_climb.json \
    --num_envs 64 --num_steps 500 --min_completion 0.5
"""

import argparse
import json
import os
import sys
import isaacgym  # 必须在 torch 前导入
import torch
import numpy as np

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_full_frame(env):
    """从环境中提取 62 维全帧, 与 MotionLoaderNing 格式一致。

    布局: [root_pos(3), root_rot(4), joint_pose(18), toe_pos(12),
           lin_vel(3), ang_vel(3), joint_vel(18), base_height(1)]
    """
    N = env.num_envs
    device = env.device

    # root_pos [N, 3] (世界坐标)
    root_pos = env.root_states[:, :3]
    # root_rot [N, 4] (四元数 wxyz or xyzw, 看框架约定)
    root_rot = env.root_states[:, 3:7]
    # joint_pose [N, 18]
    joint_pose = env.dof_pos
    # toe_pos_local [N, 12]
    if hasattr(env, "feet_pos") and env.feet_pos is not None:
        feet_flat = env.feet_pos.reshape(N, -1)
        if feet_flat.shape[1] < 12:
            pad = torch.zeros(N, 12 - feet_flat.shape[1], device=device)
            toe_pos = torch.cat([feet_flat, pad], dim=1)
        else:
            toe_pos = feet_flat[:, :12]
    else:
        toe_pos = torch.zeros(N, 12, device=device)
    # velocities
    lin_vel = env.base_lin_vel  # [N, 3]
    ang_vel = env.base_ang_vel  # [N, 3]
    joint_vel = env.dof_vel  # [N, 18]
    # base_height [N, 1]
    if hasattr(env, "measured_heights") and env.measured_heights is not None:
        terrain_h = env.measured_heights.mean(dim=1)
    else:
        terrain_h = torch.zeros(N, device=device)
    base_height = (env.root_states[:, 2] - terrain_h).unsqueeze(1)

    return torch.cat(
        [root_pos, root_rot, joint_pose, toe_pos, lin_vel, ang_vel, joint_vel, base_height],
        dim=1,
    )  # [N, 62]


def collect(args):
    from humanoid.envs import *
    from humanoid.utils.task_registry import task_registry
    from humanoid.algo.ppo.on_policy_runner import OnPolicyRunner

    # 加载环境配置
    env_cfg, train_cfg = task_registry.get_cfgs(name="n2_stairs_walk")
    env_cfg.env.num_envs = args.num_envs

    # 创建环境
    env, env_cfg = task_registry.make_env(
        name="n2_stairs_walk", args=None, env_cfg=env_cfg
    )

    # 加载策略
    train_cfg_dict = vars(train_cfg) if not isinstance(train_cfg, dict) else train_cfg
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device=env.device)

    # 加载 checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    loaded = runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    sim_dt = getattr(env, "dt", 0.02)
    all_frames = []

    obs = env.get_observations()
    episode_frames = [[] for _ in range(env.num_envs)]
    episode_completions = [0.0] * env.num_envs

    print(f"Collecting {args.num_steps} steps from {env.num_envs} envs...")

    for step in range(args.num_steps):
        # 提取全帧
        full_frame = build_full_frame(env)  # [N, 62]

        # 记录
        for i in range(env.num_envs):
            episode_frames[i].append(full_frame[i].cpu().numpy().tolist())

        # 执行一步
        with torch.inference_mode():
            actions = policy(obs.detach())
        obs, _, _, dones, infos, _, _ = env.step(actions.detach())

        # 处理终止的 episode
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_ids.cpu().numpy():
            frames = episode_frames[idx]
            if len(frames) >= 20:  # 至少 20 帧
                all_frames.extend(frames)
            episode_frames[idx] = []

        if step % 100 == 0:
            print(f"  Step {step}/{args.num_steps}, collected {len(all_frames)} frames")

    # 收集未终止的 episode
    for i in range(env.num_envs):
        if len(episode_frames[i]) >= 20:
            all_frames.extend(episode_frames[i])

    print(f"Total frames collected: {len(all_frames)}")

    if len(all_frames) == 0:
        print("ERROR: No frames collected! Check checkpoint path.")
        return

    # 保存
    out_dict = {
        "Frames": all_frames,
        "MotionWeight": 1.0,
        "FrameDuration": float(sim_dt),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_dict, f)

    print(f"Saved {len(all_frames)} frames to {args.output}")
    print(f"Frame dim: {len(all_frames[0])}, Duration: {sim_dt}s")


def main():
    parser = argparse.ArgumentParser(description="Collect reference motions for AMP")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (model_best.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="humanoid/amp_data/stair_climb.json",
        help="Output JSON path",
    )
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=500)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
