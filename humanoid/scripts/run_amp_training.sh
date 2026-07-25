#!/usr/bin/env bash
# ============================================================================
# run_amp_training.sh — AMP (Adversarial Motion Priors) 训练启动脚本
#
# 用法:
#   bash humanoid/scripts/run_amp_training.sh collect   # 仅采集参考动作
#   bash humanoid/scripts/run_amp_training.sh train     # 采集 + 训练
#   bash humanoid/scripts/run_amp_training.sh train_only # 仅训练 (已有参考数据)
#
# 环境变量 (可选):
#   N2_AMP_CHECKPOINT     - 策略 checkpoint 路径 (默认: 自动寻找最优)
#   N2_AMP_STYLE_WEIGHT   - AMP 风格奖励权重 (默认: 0.5)
#   N2_AMP_DISC_LR        - 判别器学习率 (默认: 1e-4)
#   N2_AMP_MOTION_FILE    - 参考动作 JSON 路径
#   N2_SEED               - 随机种子 (默认: 42)
#   N2_NUM_ENVS           - 训练环境数 (默认: 4096)
#   N2_AMP_COLLECT_ENVS   - 采集环境数 (默认: 64)
#   N2_AMP_COLLECT_STEPS  - 采集步数 (默认: 500)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# --- 默认参数 ---
MODE="${1:-train}"
SEED="${N2_SEED:-42}"
NUM_ENVS="${N2_NUM_ENVS:-4096}"
STYLE_WEIGHT="${N2_AMP_STYLE_WEIGHT:-0.5}"
DISC_LR="${N2_AMP_DISC_LR:-1e-4}"
REPLAY_SIZE="${N2_AMP_REPLAY_SIZE:-100000}"
MOTION_FILE="${N2_AMP_MOTION_FILE:-${PROJECT_ROOT}/humanoid/amp_data/stair_climb.json}"
COLLECT_ENVS="${N2_AMP_COLLECT_ENVS:-64}"
COLLECT_STEPS="${N2_AMP_COLLECT_STEPS:-500}"
LEARNING_RATE="${N2_AMP_LR:-1e-5}"
MAX_ITERATIONS="${N2_AMP_MAX_ITERATIONS:-10000}"
TERRAIN_MIX="${N2_AMP_TERRAIN_MIX:-0,1,2,3,4,4,4,4}"

# --- Checkpoint 自动发现 ---
if [[ -n "${N2_AMP_CHECKPOINT:-}" ]]; then
    CHECKPOINT="${N2_AMP_CHECKPOINT}"
else
    # 尝试找 stability_selected 的 best
    SELECTED="${PROJECT_ROOT}/logs/isaac_launcher/stability_selected_s${SEED}/model_best.pt"
    if [[ -f "${SELECTED}" ]]; then
        CHECKPOINT="${SELECTED}"
    else
        echo "ERROR: No checkpoint found. Set N2_AMP_CHECKPOINT." >&2
        exit 1
    fi
fi

echo "============================================"
echo " AMP Training Configuration"
echo "============================================"
echo "  Mode:           ${MODE}"
echo "  Checkpoint:     ${CHECKPOINT}"
echo "  Motion File:    ${MOTION_FILE}"
echo "  Style Weight:   ${STYLE_WEIGHT}"
echo "  Disc LR:        ${DISC_LR}"
echo "  Policy LR:      ${LEARNING_RATE}"
echo "  Seed:           ${SEED}"
echo "  Num Envs:       ${NUM_ENVS}"
echo "  Max Iterations: ${MAX_ITERATIONS}"
echo "  Terrain Mix:    ${TERRAIN_MIX}"
echo "============================================"

# --- 采集参考动作 ---
collect_motions() {
    echo ""
    echo "[AMP] Step 1: Collecting reference motions..."
    python -u humanoid/scripts/collect_reference_motions.py \
        --model_path "${CHECKPOINT}" \
        --output "${MOTION_FILE}" \
        --num_envs "${COLLECT_ENVS}" \
        --num_steps "${COLLECT_STEPS}"
    echo "[AMP] Reference motions saved to ${MOTION_FILE}"
}

# --- 训练 ---
run_training() {
    echo ""
    echo "[AMP] Step 2: Starting AMP-augmented training..."

    # 只保留任务奖励, 关闭手工风格奖励
    # (forward_progress, success, completion, termination 保留)
    STYLE_ZERO_OVERRIDES="stairs_alternating_tread=0,stairs_repeated_lead=0,stairs_same_tread_join=0"
    STYLE_ZERO_OVERRIDES+=",stairs_stride_symmetry=0,stairs_arm_swing=0"
    STYLE_ZERO_OVERRIDES+=",stairs_foothold_lateral_error=0,stairs_foot_crossover=0"
    STYLE_ZERO_OVERRIDES+=",stairs_foot_lane_error=0,stairs_single_support_stability=0"
    STYLE_ZERO_OVERRIDES+=",stairs_right_support_stability=0"

    python -u humanoid/scripts/train.py \
        --task=n2_stairs_walk \
        --resume \
        --load_run="$(dirname "${CHECKPOINT}")" \
        --checkpoint="$(basename "${CHECKPOINT}" .pt | sed 's/model_//')" \
        --headless \
        --sim_device=cuda:0 \
        --rl_device=cuda:0 \
        --num_envs="${NUM_ENVS}" \
        --max_iterations="${MAX_ITERATIONS}" \
        --seed="${SEED}" \
        --terrain_level_mix="${TERRAIN_MIX}" \
        --learning_rate="${LEARNING_RATE}" \
        --reward_scale_overrides="${STYLE_ZERO_OVERRIDES}" \
        --experiment_name=n2_stairs_amp
}

# --- 主逻辑 ---
case "${MODE}" in
    collect)
        collect_motions
        ;;
    train)
        if [[ ! -f "${MOTION_FILE}" ]]; then
            collect_motions
        else
            echo "[AMP] Using existing motion file: ${MOTION_FILE}"
        fi
        run_training
        ;;
    train_only)
        if [[ ! -f "${MOTION_FILE}" ]]; then
            echo "ERROR: Motion file not found: ${MOTION_FILE}" >&2
            echo "Run with 'collect' or 'train' mode first." >&2
            exit 1
        fi
        run_training
        ;;
    *)
        echo "Usage: $0 {collect|train|train_only}" >&2
        exit 1
        ;;
esac

echo ""
echo "[AMP] Done!"
