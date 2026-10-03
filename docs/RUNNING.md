# N2 运行说明

## 两种使用方式

1. 只回放归档策略：需要 Python、PyTorch、MuJoCo、NumPy 与 PyYAML；不需要训练 checkpoint 或完整训练日志。
2. 重新训练：另需 NVIDIA Isaac Gym Preview 4、兼容 CUDA / GPU 环境和训练依赖。

原项目记录的环境为 Ubuntu 20.04、Python 3.8、PyTorch 1.13.1；setup.py 声明 MuJoCo 2.3.6、NumPy 1.23.5 等依赖。具体环境应与所用模型和训练版本保持一致。原安装材料保留于 reference/UPSTREAM_README.md。

## 只回放

在依赖已就绪的独立 Python 环境中注册项目路径：

~~~bash
python -m pip install -e . --no-deps
~~~

先检查配置、模型与观测接口：

~~~bash
python sim2sim/sim2sim_parkour.py \
  --config_file=n2_parkour_slow_stable.yaml \
  --campaign_mode --check_only
~~~

再运行可视化：

~~~bash
python sim2sim/sim2sim_parkour.py \
  --config_file=n2_parkour_slow_stable.yaml \
  --campaign_mode --render_fps=30 --no_debug_viz
~~~

无显示环境可使用批量评测：

~~~bash
python sim2sim/evaluate_parkour_batch.py \
  sim2sim/policy/policy_best_stable.pt \
  --config_file=n2_parkour_slow_stable.yaml \
  --duration=180 --output=results/local_replay.json
~~~

输出是当前环境的复核结果，和 docs/results 中的历史记录分别保存。

## 训练与导出

~~~bash
python humanoid/scripts/train.py \
  --task=n2_parkour_slow_stable --headless --num_envs=2048 \
  --sim_device=cuda:0 --rl_device=cuda:0
~~~

训练产物写入日志目录，不纳入 Git。导出应使用对应训练配置恢复 Actor，再生成 JIT；部署侧使用推理策略，不直接使用含优化器和 Critic 的训练 checkpoint。导出入口和参数见 humanoid/scripts/play.py。

## 单元测试

~~~bash
python -m unittest discover -s tests -v
~~~

部分 MuJoCo 碰撞 / 高度场测试会在缺少可选 MuJoCo 包时自动跳过。测试涵盖课程地形、策略筛选规则、重置条件、摄像机跟随与流式显示起始控制。

## 发布的默认版本

- 部署模型：sim2sim/policy/policy_best_stable.pt。
- 观测：1370 维；动作：10 维。
- 策略周期：0.02 s；物理步长：0.002 s。
- 跑道：row 3 / seed 5 的固定五段课程。
- 控制及地形参数：以 n2_parkour_slow_stable.yaml 为准。

视频为已有实验演示，默认权重、最终历史结果和当前回放之间需按各自配置核对。
