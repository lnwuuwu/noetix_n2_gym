#!/usr/bin/env bash
# Curated AMP collection and protected N2 stair-policy refinement.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

MODE="${1:-train}"
SEED="${N2_SEED:-42}"
SIM_DEVICE="${N2_SIM_DEVICE:-cuda:0}"
RL_DEVICE="${N2_RL_DEVICE:-cuda:0}"
NUM_ENVS="${N2_NUM_ENVS:-512}"
COLLECT_ENVS="${N2_AMP_COLLECT_ENVS:-64}"
COLLECT_STEPS="${N2_AMP_COLLECT_STEPS:-6000}"
MAX_MOTIONS="${N2_AMP_MAX_MOTIONS:-24}"
COLLECT_MAX_LATERAL="${N2_AMP_MAX_LATERAL:-0.10}"
COLLECT_MAX_FINAL_LATERAL="${N2_AMP_MAX_FINAL_LATERAL:-0.04}"
COLLECT_MAX_STRIDE_IMBALANCE="${N2_AMP_MAX_STRIDE_IMBALANCE:-0.06}"
ADDITIONAL_ITERATIONS="${N2_AMP_ADDITIONAL_ITERATIONS:-200}"
STYLE_WEIGHT="${N2_AMP_STYLE_WEIGHT:-0.10}"
DISC_LR="${N2_AMP_DISC_LR:-1e-4}"
POLICY_LR="${N2_AMP_LR:-1e-6}"
REPLAY_SIZE="${N2_AMP_REPLAY_SIZE:-100000}"
AMP_BATCH_SIZE="${N2_AMP_BATCH_SIZE:-512}"
PRELOAD_TRANSITIONS="${N2_AMP_PRELOAD_TRANSITIONS:-50000}"
DISC_UPDATES="${N2_AMP_DISC_UPDATES:-2}"
GRADIENT_PENALTY="${N2_AMP_GRADIENT_PENALTY:-10.0}"
WARMUP_UPDATES="${N2_AMP_WARMUP_UPDATES:-10}"
RAMP_UPDATES="${N2_AMP_RAMP_UPDATES:-100}"
TERRAIN_MIX="${N2_AMP_TERRAIN_MIX:-0,1,2,3,4,4,4,4}"
COMMAND_SPEED="${N2_AMP_COMMAND_SPEED:-0.18}"
ACTOR_REFERENCE="${N2_AMP_ACTOR_REFERENCE:-0.60}"
SYMMETRY_COEFF="${N2_AMP_SYMMETRY_COEFF:-0.001}"
ACTION_NOISE="${N2_AMP_ACTION_NOISE:-0.03}"
ACTOR_LAYERS="${N2_AMP_ACTOR_LAYERS:-4}"
POLICY_LOSS_SCALE="${N2_AMP_POLICY_LOSS_SCALE:-1.0}"
OBSERVATION_NOISE="${N2_AMP_OBSERVATION_NOISE:-0.03}"
REWARD_OVERRIDES="${N2_AMP_REWARD_OVERRIDES:-action_rate=-0.15,action_smoothness=-0.10,dof_acc=-4e-7,stairs_lateral_drift=-12,stairs_left_drift=-4,stairs_lateral_excursion=-3,stairs_terminal_lateral=-2,stairs_heading_alignment=4,stairs_stride_symmetry=-3,stairs_right_stride_excess=-2,stairs_right_stride_excess_continuous=-3,stairs_foothold_lateral=1,stairs_foothold_lateral_error=-2,stairs_foot_crossover=-6,stairs_foot_lane_error=-4,stairs_single_support_stability=-2,stairs_right_support_stability=-3,stairs_swing_timeout=-3,stairs_alternating_tread=3,stairs_repeated_lead=-3,stairs_same_tread_join=-4}"
SAVE_INTERVAL="${N2_AMP_SAVE_INTERVAL:-20}"
# Use a new dataset namespace so an old manifest collected with the former
# 12 cm stride-imbalance gate can never be silently reused as "expert" data.
MOTION_DIR="${N2_AMP_MOTION_DIR:-${PROJECT_ROOT}/humanoid/amp_data/stair_climb_targeted_v3_s${SEED}}"
MOTION_MANIFEST="${N2_AMP_MOTION_MANIFEST:-${MOTION_DIR}/manifest.txt}"
LAUNCHER_DIR="${PROJECT_ROOT}/logs/amp_launcher"

mkdir -p "${LAUNCHER_DIR}"

if [[ "${MODE}" == "status" ]]; then
    pgrep -af '[t]rain_amp.py' || echo "No AMP trainer is running."
    if [[ -f "${LAUNCHER_DIR}/latest_amp_logpath" ]]; then
        cat "${LAUNCHER_DIR}/latest_amp_logpath"
    fi
    exit 0
fi
if [[ "${MODE}" == "log" ]]; then
    if [[ ! -f "${LAUNCHER_DIR}/latest_amp_logpath" ]]; then
        echo "No AMP training log has been recorded." >&2
        exit 2
    fi
    tail -f "$(<"${LAUNCHER_DIR}/latest_amp_logpath")"
    exit 0
fi

find_default_checkpoint() {
    local selected_dir
    local candidate
    selected_dir="${PROJECT_ROOT}/logs/isaac_launcher/stability_selected_s${SEED}"
    if [[ -f "${selected_dir}/model_best.pt" ]]; then
        printf '%s\n' "${selected_dir}/model_best.pt"
        return
    fi
    candidate="$(
        find "${selected_dir}" -maxdepth 1 -type f -name 'model_[0-9]*.pt' \
            -print 2>/dev/null | sort -V | tail -n 1
    )"
    if [[ -n "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return
    fi
    return 1
}

if [[ -n "${N2_AMP_CHECKPOINT:-}" ]]; then
    CHECKPOINT="${N2_AMP_CHECKPOINT}"
else
    CHECKPOINT="$(find_default_checkpoint || true)"
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: AMP base checkpoint not found." >&2
    echo "Set N2_AMP_CHECKPOINT=/absolute/path/to/model_N.pt" >&2
    exit 1
fi
CHECKPOINT="$(readlink -f "${CHECKPOINT}")"

checkpoint_iteration() {
    python -c 'import sys, torch
path = sys.argv[1]
try:
    data = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    data = torch.load(path, map_location="cpu")
iteration = int(data.get("iter", -1))
if iteration < 0:
    raise SystemExit("checkpoint has no valid iter metadata: " + path)
print(iteration)' "${CHECKPOINT}"
}

CURRENT_ITERATION="$(checkpoint_iteration)"
if [[ -n "${N2_AMP_MAX_ITERATIONS:-}" ]]; then
    MAX_ITERATIONS="${N2_AMP_MAX_ITERATIONS}"
else
    MAX_ITERATIONS="$((CURRENT_ITERATION + ADDITIONAL_ITERATIONS))"
fi

collect_motions() {
    mkdir -p "${MOTION_DIR}"
    python -u humanoid/scripts/collect_reference_motions.py \
        --task=n2_stairs_walk \
        --model_path="${CHECKPOINT}" \
        --output_dir="${MOTION_DIR}" \
        --manifest="${MOTION_MANIFEST}" \
        --headless \
        --sim_device="${SIM_DEVICE}" \
        --rl_device="${RL_DEVICE}" \
        --num_envs="${COLLECT_ENVS}" \
        --num_steps="${COLLECT_STEPS}" \
        --max_motions="${MAX_MOTIONS}" \
        --max_lateral_deviation="${COLLECT_MAX_LATERAL}" \
        --max_final_lateral_position="${COLLECT_MAX_FINAL_LATERAL}" \
        --max_stride_imbalance="${COLLECT_MAX_STRIDE_IMBALANCE}" \
        --fixed_terrain_level=4 \
        --command_speed="${COMMAND_SPEED}" \
        --seed="${SEED}"
}

run_training() {
    local log_path
    local run_name
    local run_dir
    local selected_dir
    if [[ ! -f "${MOTION_MANIFEST}" ]]; then
        echo "ERROR: motion manifest not found: ${MOTION_MANIFEST}" >&2
        exit 1
    fi
    if (( MAX_ITERATIONS <= CURRENT_ITERATION )); then
        echo "ERROR: target iteration ${MAX_ITERATIONS} is not above checkpoint iteration ${CURRENT_ITERATION}." >&2
        exit 1
    fi

    run_name="amp_curated_from_${CURRENT_ITERATION}_to_${MAX_ITERATIONS}_s${SEED}"
    log_path="${LAUNCHER_DIR}/${run_name}_$(date +%m%d_%H-%M-%S).log"
    printf '%s\n' "${log_path}" > "${LAUNCHER_DIR}/latest_amp_logpath"
    echo "[AMP] log: ${log_path}"

    python -u humanoid/scripts/train_amp.py \
        --task=n2_stairs_walk \
        --resume \
        --model_path="${CHECKPOINT}" \
        --motion_manifest="${MOTION_MANIFEST}" \
        --headless \
        --sim_device="${SIM_DEVICE}" \
        --rl_device="${RL_DEVICE}" \
        --num_envs="${NUM_ENVS}" \
        --max_iterations="${MAX_ITERATIONS}" \
        --seed="${SEED}" \
        --terrain_level_mix="${TERRAIN_MIX}" \
        --command_speed="${COMMAND_SPEED}" \
        --learning_rate="${POLICY_LR}" \
        --fixed_learning_rate \
        --reset_optimizer \
        --action_noise_std="${ACTION_NOISE}" \
        --actor_reference_loss_coeff="${ACTOR_REFERENCE}" \
        --actor_policy_loss_scale="${POLICY_LOSS_SCALE}" \
        --actor_trainable_layers="${ACTOR_LAYERS}" \
        --freeze_action_noise \
        --symmetry_loss_coeff="${SYMMETRY_COEFF}" \
        --observation_noise_level="${OBSERVATION_NOISE}" \
        --reward_scale_overrides="${REWARD_OVERRIDES}" \
        --save_interval="${SAVE_INTERVAL}" \
        --amp_style_weight="${STYLE_WEIGHT}" \
        --amp_disc_lr="${DISC_LR}" \
        --amp_replay_size="${REPLAY_SIZE}" \
        --amp_batch_size="${AMP_BATCH_SIZE}" \
        --amp_preload_transitions="${PRELOAD_TRANSITIONS}" \
        --amp_disc_updates="${DISC_UPDATES}" \
        --amp_gradient_penalty="${GRADIENT_PENALTY}" \
        --amp_reward_warmup_updates="${WARMUP_UPDATES}" \
        --amp_reward_ramp_updates="${RAMP_UPDATES}" \
        --experiment_name=n2_stairs_amp \
        --run_name="${run_name}" 2>&1 | tee "${log_path}"

    run_dir="$(
        find "${PROJECT_ROOT}/logs/n2_stairs_amp" \
            -mindepth 1 -maxdepth 1 -type d \
            -name "*_${run_name}" -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -n 1 | cut -d' ' -f2-
    )"
    if [[ -z "${run_dir}" || ! -f "${run_dir}/model_best.pt" ]]; then
        echo "ERROR: AMP run did not produce model_best.pt" >&2
        exit 1
    fi
    selected_dir="${LAUNCHER_DIR}/selected_s${SEED}"
    mkdir -p "${selected_dir}"
    cp -f "${run_dir}/model_best.pt" "${selected_dir}/model_best.pt"
    printf '%s\n' "${run_dir}" > "${selected_dir}/run_dir.txt"
    printf '%s\n' "${run_dir}/model_best.pt" \
        > "${selected_dir}/source_checkpoint.txt"
    echo "N2_AMP_RUN=${run_dir}"
    echo "N2_AMP_BEST=${selected_dir}/model_best.pt"
}

print_configuration() {
    echo "AMP mode=${MODE}"
    echo "checkpoint=${CHECKPOINT} (iteration ${CURRENT_ITERATION})"
    echo "motion_manifest=${MOTION_MANIFEST}"
    echo "motion_gate=max_lateral=${COLLECT_MAX_LATERAL} final_lateral=${COLLECT_MAX_FINAL_LATERAL} stride_imbalance=${COLLECT_MAX_STRIDE_IMBALANCE}"
    echo "target_iteration=${MAX_ITERATIONS}"
    echo "envs=${NUM_ENVS} terrain_mix=${TERRAIN_MIX}"
    echo "policy_lr=${POLICY_LR} style_weight=${STYLE_WEIGHT} actor_layers=${ACTOR_LAYERS}"
    echo "observation_noise=${OBSERVATION_NOISE}"
    echo "targeted_reward_overrides=${REWARD_OVERRIDES}"
}

case "${MODE}" in
    collect)
        print_configuration
        collect_motions
        ;;
    train)
        print_configuration
        if [[ ! -f "${MOTION_MANIFEST}" ]]; then
            collect_motions
        else
            echo "[AMP] Reusing curated manifest: ${MOTION_MANIFEST}"
        fi
        run_training
        ;;
    train_only)
        print_configuration
        run_training
        ;;
    smoke)
        NUM_ENVS="${N2_NUM_ENVS:-64}"
        PRELOAD_TRANSITIONS="${N2_AMP_PRELOAD_TRANSITIONS:-2048}"
        MAX_ITERATIONS="$((CURRENT_ITERATION + 2))"
        print_configuration
        run_training
        ;;
    *)
        echo "Usage: $0 {collect|train|train_only|smoke|status|log}" >&2
        exit 2
        ;;
esac
