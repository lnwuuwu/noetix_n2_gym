#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-}"
TRAIN_SEED="${N2_SEED:-42}"
TRAIN_DEVICE="${N2_DEVICE:-cuda:0}"
TRAIN_ENVS="${N2_NUM_ENVS:-256}"
INIT_CHECKPOINT="${N2_INIT_CHECKPOINT:-}"
PILOT_ITERATIONS="${N2_PILOT_ITERATIONS:-50}"
LEARNING_RATE="${N2_LEARNING_RATE:-5e-6}"
ACTION_NOISE_STD="${N2_ACTION_NOISE_STD:-0.10}"
REFERENCE_COEFF="${N2_REFERENCE_COEFF:-0.25}"
SYMMETRY_COEFF="${N2_SYMMETRY_COEFF:-0.05}"
STREAM_PORT="${N2_STREAM_PORT:-8080}"
CAMERA_WIDTH="${N2_CAMERA_WIDTH:-960}"
CAMERA_HEIGHT="${N2_CAMERA_HEIGHT:-540}"
JPEG_QUALITY="${N2_JPEG_QUALITY:-80}"
VIEW_CHECKPOINT="${N2_VIEW_CHECKPOINT:-}"
LAUNCHER_DIR="${ROOT_DIR}/logs/isaac_launcher"
PID_FILE="${LAUNCHER_DIR}/n2_stairs_polish_s${TRAIN_SEED}.pid"
ACTIVE_LOG_FILE="${LAUNCHER_DIR}/n2_stairs_polish_s${TRAIN_SEED}.logpath"

mkdir -p "${LAUNCHER_DIR}"

usage() {
    echo "Usage: $0 baseline|smoke|pilot|status|log|stop|candidate|compare|view"
    echo "Set N2_INIT_CHECKPOINT=/absolute/path/model_9000.pt for train/eval modes."
    echo "Set N2_VIEW_CHECKPOINT to override automatic model_9050.pt discovery."
}

require_checkpoint() {
    if [[ -z "${INIT_CHECKPOINT}" ]]; then
        echo "N2_INIT_CHECKPOINT is required." >&2
        exit 2
    fi
    if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
        echo "Checkpoint does not exist: ${INIT_CHECKPOINT}" >&2
        exit 2
    fi
    INIT_CHECKPOINT="$(readlink -f "${INIT_CHECKPOINT}")"
    CHECKPOINT_NAME="$(basename "${INIT_CHECKPOINT}")"
    if [[ ! "${CHECKPOINT_NAME}" =~ ^model_([0-9]+)\.pt$ ]]; then
        echo "Checkpoint must be named model_<iteration>.pt: ${INIT_CHECKPOINT}" >&2
        exit 2
    fi
    CHECKPOINT_ITERATION="${BASH_REMATCH[1]}"
    CHECKPOINT_RUN="$(dirname "${INIT_CHECKPOINT}")"
    PILOT_TARGET=$((CHECKPOINT_ITERATION + PILOT_ITERATIONS))
    RUN_NAME="isaac_l4_guarded_pilot_from_${CHECKPOINT_ITERATION}_s${TRAIN_SEED}"
}

active_pid() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(<"${PID_FILE}")"
        if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
            echo "${pid}"
            return 0
        fi
    fi
    return 1
}

run_train() {
    local extra_iterations="$1"
    local foreground="$2"
    local target=$((CHECKPOINT_ITERATION + extra_iterations))
    local suffix="pilot"
    if [[ "${foreground}" == "yes" ]]; then
        suffix="smoke"
    fi
    local run_name="isaac_l4_guarded_${suffix}_from_${CHECKPOINT_ITERATION}_s${TRAIN_SEED}"
    local timestamp
    timestamp="$(date +%m%d_%H-%M-%S)"
    local log_path="${LAUNCHER_DIR}/${run_name}_${timestamp}.log"
    local command=(
        python -u humanoid/scripts/train.py
        --task=n2_stairs_walk
        --resume
        "--load_run=${CHECKPOINT_RUN}"
        "--checkpoint=${CHECKPOINT_ITERATION}"
        --headless
        "--sim_device=${TRAIN_DEVICE}"
        "--rl_device=${TRAIN_DEVICE}"
        "--num_envs=${TRAIN_ENVS}"
        "--max_iterations=${target}"
        "--run_name=${run_name}"
        "--seed=${TRAIN_SEED}"
        --fixed_terrain_level=4
        --command_speed=0.18
        --reset_optimizer
        "--learning_rate=${LEARNING_RATE}"
        --fixed_learning_rate
        "--action_noise_std=${ACTION_NOISE_STD}"
        --freeze_action_noise
        --actor_head_only
        "--actor_reference_loss_coeff=${REFERENCE_COEFF}"
        "--symmetry_loss_coeff=${SYMMETRY_COEFF}"
    )

    printf '%s\n' "${log_path}" > "${ACTIVE_LOG_FILE}"
    echo "Checkpoint: ${INIT_CHECKPOINT}"
    echo "Iterations: ${CHECKPOINT_ITERATION} -> ${target}"
    echo "Log: ${log_path}"
    if [[ "${foreground}" == "yes" ]]; then
        "${command[@]}" 2>&1 | tee "${log_path}"
    else
        if active_pid >/dev/null; then
            echo "A guarded Isaac Gym trainer is already running (PID $(active_pid))." >&2
            exit 2
        fi
        nohup "${command[@]}" > "${log_path}" 2>&1 &
        local pid=$!
        printf '%s\n' "${pid}" > "${PID_FILE}"
        echo "Started guarded Isaac Gym pilot PID=${pid}"
        echo "Watch: $0 log"
    fi
}

eval_checkpoint() {
    local checkpoint_path="$1"
    local label="$2"
    local checkpoint_name
    local checkpoint_iteration
    local checkpoint_run
    checkpoint_path="$(readlink -f "${checkpoint_path}")"
    checkpoint_name="$(basename "${checkpoint_path}")"
    if [[ ! "${checkpoint_name}" =~ ^model_([0-9]+)\.pt$ ]]; then
        echo "Cannot evaluate nonstandard checkpoint: ${checkpoint_path}" >&2
        exit 2
    fi
    checkpoint_iteration="${BASH_REMATCH[1]}"
    checkpoint_run="$(dirname "${checkpoint_path}")"
    local output_dir="${LAUNCHER_DIR}/evaluations"
    local output="${output_dir}/${label}_s${TRAIN_SEED}.csv"
    mkdir -p "${output_dir}"
    python -u humanoid/scripts/eval_stairs.py \
        --task=n2_stairs_walk \
        --resume \
        "--load_run=${checkpoint_run}" \
        "--checkpoint=${checkpoint_iteration}" \
        --headless \
        "--sim_device=${TRAIN_DEVICE}" \
        "--rl_device=${TRAIN_DEVICE}" \
        --num_envs=128 \
        "--seed=${TRAIN_SEED}" \
        --terrain_levels=4 \
        --command_speed=0.18 \
        --episodes_per_env=1 \
        "--output=${output}"
    echo "Evaluation: ${output}"
}

latest_candidate() {
    local matches=()
    shopt -s nullglob
    matches=(
        "${ROOT_DIR}/logs/n2_stairs_walk/"*"_${RUN_NAME}"
    )
    shopt -u nullglob
    if [[ "${#matches[@]}" -eq 0 ]]; then
        echo "No pilot run found for ${RUN_NAME}" >&2
        return 1
    fi
    local newest="${matches[0]}"
    local run
    for run in "${matches[@]:1}"; do
        if [[ "${run}" -nt "${newest}" ]]; then
            newest="${run}"
        fi
    done
    local candidate="${newest}/model_${PILOT_TARGET}.pt"
    if [[ ! -f "${candidate}" ]]; then
        echo "Pilot checkpoint is not ready: ${candidate}" >&2
        return 1
    fi
    echo "${candidate}"
}

resolve_view_checkpoint() {
    if [[ -n "${VIEW_CHECKPOINT}" ]]; then
        if [[ ! -f "${VIEW_CHECKPOINT}" ]]; then
            echo "View checkpoint does not exist: ${VIEW_CHECKPOINT}" >&2
            return 1
        fi
        readlink -f "${VIEW_CHECKPOINT}"
        return
    fi

    local selected="${HOME}/n2_checkpoints/isaac_9050/model_9050.pt"
    if [[ -f "${selected}" ]]; then
        readlink -f "${selected}"
        return
    fi

    local matches=()
    shopt -s nullglob
    matches=(
        "${ROOT_DIR}/logs/n2_stairs_walk/"*"_isaac_l4_guarded_pilot_from_9000_s${TRAIN_SEED}/model_9050.pt"
    )
    shopt -u nullglob
    if [[ "${#matches[@]}" -eq 0 ]]; then
        echo "Cannot find model_9050.pt; set N2_VIEW_CHECKPOINT explicitly." >&2
        return 1
    fi
    local newest="${matches[0]}"
    local checkpoint
    for checkpoint in "${matches[@]:1}"; do
        if [[ "${checkpoint}" -nt "${newest}" ]]; then
            newest="${checkpoint}"
        fi
    done
    readlink -f "${newest}"
}

stream_checkpoint() {
    local checkpoint_path="$1"
    local checkpoint_name
    local checkpoint_iteration
    local checkpoint_run
    checkpoint_name="$(basename "${checkpoint_path}")"
    if [[ ! "${checkpoint_name}" =~ ^model_([0-9]+)\.pt$ ]]; then
        echo "Cannot stream nonstandard checkpoint: ${checkpoint_path}" >&2
        exit 2
    fi
    checkpoint_iteration="${BASH_REMATCH[1]}"
    checkpoint_run="$(dirname "${checkpoint_path}")"
    if ! python -c "from PIL import Image" >/dev/null 2>&1; then
        echo "Pillow is missing. Install it with: python -m pip install Pillow" >&2
        exit 2
    fi
    echo "Streaming checkpoint: ${checkpoint_path}"
    echo "Server endpoint: http://127.0.0.1:${STREAM_PORT}/"
    echo "This is off-screen rendering; no server GUI will be created."
    python -u humanoid/scripts/stream_stairs.py \
        --task=n2_stairs_walk \
        --resume \
        "--load_run=${checkpoint_run}" \
        "--checkpoint=${checkpoint_iteration}" \
        --headless \
        "--sim_device=${TRAIN_DEVICE}" \
        "--rl_device=${TRAIN_DEVICE}" \
        "--seed=${TRAIN_SEED}" \
        --terrain_level=4 \
        --command_speed=0.18 \
        "--stream_port=${STREAM_PORT}" \
        "--camera_width=${CAMERA_WIDTH}" \
        "--camera_height=${CAMERA_HEIGHT}" \
        "--jpeg_quality=${JPEG_QUALITY}"
}

case "${MODE}" in
    baseline)
        require_checkpoint
        eval_checkpoint "${INIT_CHECKPOINT}" \
            "baseline_${CHECKPOINT_ITERATION}"
        ;;
    smoke)
        require_checkpoint
        run_train 2 yes
        ;;
    pilot)
        require_checkpoint
        run_train "${PILOT_ITERATIONS}" no
        ;;
    status)
        if pid="$(active_pid)"; then
            echo "Guarded Isaac Gym pilot is running PID=${pid}"
        else
            echo "No guarded Isaac Gym pilot is running."
        fi
        if [[ -f "${ACTIVE_LOG_FILE}" ]]; then
            log_path="$(<"${ACTIVE_LOG_FILE}")"
            echo "Latest log: ${log_path}"
            [[ -f "${log_path}" ]] && tail -n 35 "${log_path}"
        fi
        ;;
    log)
        if [[ ! -f "${ACTIVE_LOG_FILE}" ]]; then
            echo "No launcher log has been recorded." >&2
            exit 2
        fi
        log_path="$(<"${ACTIVE_LOG_FILE}")"
        echo "Following ${log_path}"
        tail -f "${log_path}"
        ;;
    stop)
        if pid="$(active_pid)"; then
            kill -TERM "${pid}"
            echo "Sent SIGTERM to PID=${pid}; the trainer will save a checkpoint."
        else
            echo "No guarded Isaac Gym pilot is running."
        fi
        ;;
    candidate)
        require_checkpoint
        candidate_path="$(latest_candidate)"
        eval_checkpoint "${candidate_path}" \
            "candidate_${PILOT_TARGET}"
        ;;
    compare)
        require_checkpoint
        eval_checkpoint "${INIT_CHECKPOINT}" \
            "baseline_${CHECKPOINT_ITERATION}"
        candidate_path="$(latest_candidate)"
        eval_checkpoint "${candidate_path}" \
            "candidate_${PILOT_TARGET}"
        echo
        echo "Baseline:"
        cat "${LAUNCHER_DIR}/evaluations/baseline_${CHECKPOINT_ITERATION}_s${TRAIN_SEED}.csv"
        echo
        echo "Candidate:"
        cat "${LAUNCHER_DIR}/evaluations/candidate_${PILOT_TARGET}_s${TRAIN_SEED}.csv"
        ;;
    view)
        view_checkpoint="$(resolve_view_checkpoint)"
        stream_checkpoint "${view_checkpoint}"
        ;;
    *)
        usage
        exit 2
        ;;
esac
