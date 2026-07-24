#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-}"
TRAIN_SEED="${N2_SEED:-42}"
TRAIN_DEVICE="${N2_DEVICE:-cuda:0}"
TRAIN_ENVS="${N2_NUM_ENVS:-256}"
INIT_CHECKPOINT="${N2_INIT_CHECKPOINT:-}"
STAGE_ITERATIONS="${N2_STAGE_ITERATIONS:-25}"
STAGE_CYCLES="${N2_STAGE_CYCLES:-2}"
EVAL_ENVS="${N2_STAGE_EVAL_ENVS:-64}"
FINAL_EVAL_ENVS="${N2_FINAL_EVAL_ENVS:-128}"
LEARNING_RATE="${N2_STABILITY_LEARNING_RATE:-2e-6}"
ACTION_NOISE_STD="${N2_STABILITY_ACTION_NOISE_STD:-0.06}"
REFERENCE_COEFF="${N2_STABILITY_REFERENCE_COEFF:-0.50}"
SYMMETRY_COEFF="${N2_STABILITY_SYMMETRY_COEFF:-0.15}"
OBSERVATION_NOISE="${N2_STABILITY_OBSERVATION_NOISE:-0.20}"
REWARD_OVERRIDES="${N2_STABILITY_REWARD_OVERRIDES:-action_rate=-0.35,action_smoothness=-0.20,dof_acc=-4e-7,stairs_lateral_drift=-18,stairs_heading_alignment=4,stairs_foothold_lateral=1,stairs_foothold_lateral_error=-2,stairs_stride_symmetry=-4}"

LAUNCHER_DIR="${ROOT_DIR}/logs/isaac_launcher"
TRAIN_ROOT="${ROOT_DIR}/logs/n2_stairs_stability"
PID_FILE="${LAUNCHER_DIR}/n2_stability_curriculum_s${TRAIN_SEED}.pid"
ACTIVE_LOG_FILE="${LAUNCHER_DIR}/n2_stability_curriculum_s${TRAIN_SEED}.logpath"
WORK_ROOT="${LAUNCHER_DIR}/stability_curriculum_s${TRAIN_SEED}"
RESULT_DIR="${LAUNCHER_DIR}/stability_selected_s${TRAIN_SEED}"

mkdir -p "${LAUNCHER_DIR}" "${WORK_ROOT}" "${RESULT_DIR}"

CHILD_PID=""

usage() {
    echo "Usage: $0 smoke|long|status|log|stop"
    echo "Set N2_INIT_CHECKPOINT=/absolute/path/model_9050.pt."
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, received ${value}" >&2
        exit 2
    fi
}

require_checkpoint() {
    if [[ -z "${INIT_CHECKPOINT}" || ! -f "${INIT_CHECKPOINT}" ]]; then
        echo "N2_INIT_CHECKPOINT must name an existing checkpoint." >&2
        exit 2
    fi
    INIT_CHECKPOINT="$(readlink -f "${INIT_CHECKPOINT}")"
    local checkpoint_name
    checkpoint_name="$(basename "${INIT_CHECKPOINT}")"
    if [[ ! "${checkpoint_name}" =~ ^model_([0-9]+)\.pt$ ]]; then
        echo "Checkpoint must be named model_<iteration>.pt." >&2
        exit 2
    fi
}

checkpoint_iteration() {
    local name
    name="$(basename "$1")"
    if [[ ! "${name}" =~ ^model_([0-9]+)\.pt$ ]]; then
        echo "Cannot read checkpoint iteration: $1" >&2
        return 1
    fi
    echo "${BASH_REMATCH[1]}"
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

run_child() {
    "$@" &
    CHILD_PID=$!
    set +e
    wait "${CHILD_PID}"
    local status=$?
    set -e
    CHILD_PID=""
    return "${status}"
}

terminate_driver() {
    echo "Received stop signal; forwarding it to the active Isaac process."
    if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
        kill -TERM "${CHILD_PID}" 2>/dev/null || true
        wait "${CHILD_PID}" 2>/dev/null || true
    fi
    exit 143
}

cleanup_driver() {
    rm -f "${PID_FILE}"
}

latest_run_directory() {
    local run_name="$1"
    local newest
    newest="$(
        find "${TRAIN_ROOT}" -mindepth 1 -maxdepth 1 -type d \
            -name "*_${run_name}" -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -n 1 | cut -d' ' -f2-
    )"
    if [[ -z "${newest}" || ! -d "${newest}" ]]; then
        echo "Cannot find output run for ${run_name}" >&2
        return 1
    fi
    echo "${newest}"
}

train_stage() {
    local source_checkpoint="$1"
    local level="$2"
    local speed="$3"
    local iterations="$4"
    local cycle="$5"
    local source_iteration
    local target_iteration
    local source_run
    local run_name
    local output_run

    source_iteration="$(checkpoint_iteration "${source_checkpoint}")"
    target_iteration=$((source_iteration + iterations))
    source_run="$(dirname "${source_checkpoint}")"
    run_name="stability_c${cycle}_l${level}_from_${source_iteration}_s${TRAIN_SEED}"

    echo "ISAAC_STABILITY_TRAIN cycle=${cycle} level=${level} speed=${speed} checkpoint=${source_checkpoint}"
    run_child python -u humanoid/scripts/train.py \
        --task=n2_stairs_walk \
        --resume \
        "--load_run=${source_run}" \
        "--checkpoint=${source_iteration}" \
        --headless \
        "--sim_device=${TRAIN_DEVICE}" \
        "--rl_device=${TRAIN_DEVICE}" \
        "--num_envs=${TRAIN_ENVS}" \
        "--max_iterations=${target_iteration}" \
        --experiment_name=n2_stairs_stability \
        "--run_name=${run_name}" \
        "--seed=${TRAIN_SEED}" \
        "--fixed_terrain_level=${level}" \
        "--command_speed=${speed}" \
        --reset_optimizer \
        "--learning_rate=${LEARNING_RATE}" \
        --fixed_learning_rate \
        "--action_noise_std=${ACTION_NOISE_STD}" \
        --freeze_action_noise \
        --actor_trainable_layers=2 \
        "--actor_reference_loss_coeff=${REFERENCE_COEFF}" \
        "--symmetry_loss_coeff=${SYMMETRY_COEFF}" \
        "--observation_noise_level=${OBSERVATION_NOISE}" \
        "--reward_scale_overrides=${REWARD_OVERRIDES}"

    output_run="$(latest_run_directory "${run_name}")"
    CANDIDATE_CHECKPOINT="${output_run}/model_${target_iteration}.pt"
    if [[ ! -f "${CANDIDATE_CHECKPOINT}" ]]; then
        echo "Stage checkpoint was not saved: ${CANDIDATE_CHECKPOINT}" >&2
        return 1
    fi
}

evaluate_checkpoint() {
    local checkpoint="$1"
    local level="$2"
    local speed="$3"
    local output="$4"
    local env_count="$5"
    local iteration
    local run
    iteration="$(checkpoint_iteration "${checkpoint}")"
    run="$(dirname "${checkpoint}")"
    run_child python -u humanoid/scripts/eval_stairs.py \
        --task=n2_stairs_walk \
        --resume \
        "--load_run=${run}" \
        "--checkpoint=${iteration}" \
        --headless \
        "--sim_device=${TRAIN_DEVICE}" \
        "--rl_device=${TRAIN_DEVICE}" \
        "--num_envs=${env_count}" \
        "--seed=${TRAIN_SEED}" \
        "--terrain_levels=${level}" \
        "--command_speed=${speed}" \
        --episodes_per_env=1 \
        "--output=${output}"
}

run_curriculum() {
    require_checkpoint
    require_positive_integer N2_STAGE_ITERATIONS "${STAGE_ITERATIONS}"
    require_positive_integer N2_STAGE_CYCLES "${STAGE_CYCLES}"
    require_positive_integer N2_STAGE_EVAL_ENVS "${EVAL_ENVS}"

    trap terminate_driver TERM INT
    trap cleanup_driver EXIT

    local timestamp
    local work_dir
    local selected_checkpoint
    local current_high_csv
    local cycle
    local level
    local speed
    local prefix
    local baseline_stage_csv
    local candidate_stage_csv
    local candidate_high_csv
    local decision_json
    local gate_status
    local selected_iteration
    local selected_copy
    local final_csv
    local speeds=(0.12 0.14 0.16 0.17 0.18)

    timestamp="$(date +%m%d_%H-%M-%S)"
    work_dir="${WORK_ROOT}/${timestamp}"
    mkdir -p "${work_dir}"
    selected_checkpoint="${INIT_CHECKPOINT}"
    current_high_csv="${work_dir}/initial_10cm.csv"

    echo "ISAAC_STABILITY_START checkpoint=${selected_checkpoint}"
    echo "ISAAC_STABILITY_PLAN cycles=${STAGE_CYCLES} iterations_per_stage=${STAGE_ITERATIONS} levels=0,1,2,3,4"
    evaluate_checkpoint \
        "${selected_checkpoint}" 4 0.18 "${current_high_csv}" "${EVAL_ENVS}"

    for ((cycle = 1; cycle <= STAGE_CYCLES; cycle++)); do
        for level in 0 1 2 3 4; do
            speed="${speeds[${level}]}"
            prefix="${work_dir}/c${cycle}_l${level}"
            baseline_stage_csv="${prefix}_baseline.csv"
            candidate_stage_csv="${prefix}_candidate.csv"
            candidate_high_csv="${prefix}_candidate_10cm.csv"
            decision_json="${prefix}_decision.json"

            if [[ "${level}" -eq 4 ]]; then
                baseline_stage_csv="${current_high_csv}"
            else
                evaluate_checkpoint \
                    "${selected_checkpoint}" "${level}" "${speed}" \
                    "${baseline_stage_csv}" "${EVAL_ENVS}"
            fi

            train_stage \
                "${selected_checkpoint}" "${level}" "${speed}" \
                "${STAGE_ITERATIONS}" "${cycle}"

            evaluate_checkpoint \
                "${CANDIDATE_CHECKPOINT}" "${level}" "${speed}" \
                "${candidate_stage_csv}" "${EVAL_ENVS}"
            if [[ "${level}" -eq 4 ]]; then
                candidate_high_csv="${candidate_stage_csv}"
            else
                evaluate_checkpoint \
                    "${CANDIDATE_CHECKPOINT}" 4 0.18 \
                    "${candidate_high_csv}" "${EVAL_ENVS}"
            fi

            set +e
            python humanoid/scripts/isaac_stairs_stage_gate.py \
                "--baseline-stage=${baseline_stage_csv}" \
                "--candidate-stage=${candidate_stage_csv}" \
                "--baseline-high=${current_high_csv}" \
                "--candidate-high=${candidate_high_csv}" \
                "--stage-level=${level}" \
                "--output=${decision_json}"
            gate_status=$?
            set -e

            if [[ "${gate_status}" -eq 0 ]]; then
                selected_checkpoint="${CANDIDATE_CHECKPOINT}"
                current_high_csv="${candidate_high_csv}"
                echo "ISAAC_STABILITY_ACCEPT cycle=${cycle} level=${level} checkpoint=${selected_checkpoint}"
            else
                echo "ISAAC_STABILITY_REJECT cycle=${cycle} level=${level} keeping=${selected_checkpoint}"
            fi
        done
    done

    selected_iteration="$(checkpoint_iteration "${selected_checkpoint}")"
    selected_copy="${RESULT_DIR}/model_${selected_iteration}.pt"
    cp -f "${selected_checkpoint}" "${selected_copy}"
    printf '%s\n' "${selected_copy}" > "${RESULT_DIR}/selected_checkpoint.txt"

    final_csv="${RESULT_DIR}/evaluation_all_levels.csv"
    evaluate_checkpoint \
        "${selected_copy}" 0,1,2,3,4 0.18 \
        "${final_csv}" "${FINAL_EVAL_ENVS}"
    echo "N2_ISAAC_STABILITY_CHECKPOINT=${selected_copy}"
    echo "N2_ISAAC_STABILITY_EVALUATION=${final_csv}"
}

smoke_train() {
    require_checkpoint
    train_stage "${INIT_CHECKPOINT}" 0 0.12 2 smoke
    echo "N2_ISAAC_STABILITY_SMOKE_CHECKPOINT=${CANDIDATE_CHECKPOINT}"
}

case "${MODE}" in
    smoke)
        smoke_train
        ;;
    long)
        require_checkpoint
        require_positive_integer N2_STAGE_ITERATIONS "${STAGE_ITERATIONS}"
        require_positive_integer N2_STAGE_CYCLES "${STAGE_CYCLES}"
        if active_pid >/dev/null; then
            echo "Isaac stability curriculum is already running PID=$(active_pid)." >&2
            exit 2
        fi
        timestamp="$(date +%m%d_%H-%M-%S)"
        log_path="${LAUNCHER_DIR}/isaac_stability_curriculum_s${TRAIN_SEED}_${timestamp}.log"
        printf '%s\n' "${log_path}" > "${ACTIVE_LOG_FILE}"
        nohup bash "$0" _run > "${log_path}" 2>&1 &
        pid=$!
        printf '%s\n' "${pid}" > "${PID_FILE}"
        echo "Started Isaac stability curriculum PID=${pid}"
        echo "Log: ${log_path}"
        echo "Watch: $0 log"
        ;;
    _run)
        run_curriculum
        ;;
    status)
        if pid="$(active_pid)"; then
            echo "Isaac stability curriculum is running PID=${pid}"
        else
            echo "No Isaac stability curriculum is running."
        fi
        if [[ -f "${ACTIVE_LOG_FILE}" ]]; then
            log_path="$(<"${ACTIVE_LOG_FILE}")"
            echo "Latest log: ${log_path}"
            [[ -f "${log_path}" ]] && tail -n 40 "${log_path}"
        fi
        ;;
    log)
        if [[ ! -f "${ACTIVE_LOG_FILE}" ]]; then
            echo "No stability curriculum log has been recorded." >&2
            exit 2
        fi
        log_path="$(<"${ACTIVE_LOG_FILE}")"
        echo "Following ${log_path}"
        tail -f "${log_path}"
        ;;
    stop)
        if pid="$(active_pid)"; then
            kill -TERM "${pid}"
            echo "Sent SIGTERM to stability curriculum PID=${pid}."
        else
            echo "No Isaac stability curriculum is running."
        fi
        ;;
    *)
        usage
        exit 2
        ;;
esac
