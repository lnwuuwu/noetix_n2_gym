#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-}"
TRAIN_SEED="${N2_SEED:-42}"
TRAIN_DEVICE="${N2_DEVICE:-cuda:0}"
TRAIN_ENVS="${N2_NUM_ENVS:-256}"
# A dedicated override prevents stale experiment variables from selecting an
# unintended checkpoint.
INIT_CHECKPOINT="${N2_STABILITY_INIT_CHECKPOINT:-}"
MICRO_ITERATIONS="${N2_STABILITY_MICRO_ITERATIONS:-8}"
SEARCH_ROUNDS="${N2_STABILITY_SEARCH_ROUNDS:-2}"
EVAL_ENVS="${N2_STABILITY_EVAL_ENVS:-128}"
HOLDOUT_ENVS="${N2_STABILITY_HOLDOUT_ENVS:-256}"
TERRAIN_MIX="${N2_STABILITY_TERRAIN_MIX:-0,1,2,3,4,4,4,4}"
COMMAND_SPEED="${N2_STABILITY_COMMAND_SPEED:-0.18}"

LAUNCHER_DIR="${ROOT_DIR}/logs/isaac_launcher"
TRAIN_ROOT="${ROOT_DIR}/logs/n2_stairs_stability"
PID_FILE="${LAUNCHER_DIR}/n2_stability_search_s${TRAIN_SEED}.pid"
ACTIVE_LOG_FILE="${LAUNCHER_DIR}/n2_stability_search_s${TRAIN_SEED}.logpath"
WORK_ROOT="${LAUNCHER_DIR}/stability_search_s${TRAIN_SEED}"
RESULT_DIR="${LAUNCHER_DIR}/stability_selected_s${TRAIN_SEED}"

mkdir -p "${LAUNCHER_DIR}" "${WORK_ROOT}" "${RESULT_DIR}"

CHILD_PID=""
CANDIDATE_CHECKPOINT=""
PROFILE_LEARNING_RATE=""
PROFILE_ACTION_NOISE=""
PROFILE_REFERENCE_COEFF=""
PROFILE_SYMMETRY_COEFF=""
PROFILE_OBSERVATION_NOISE=""
PROFILE_REWARD_OVERRIDES=""

usage() {
    echo "Usage: $0 smoke|pilot|long|status|log|stop"
    echo "The latest guarded model_9050.pt is selected automatically."
    echo "Set N2_STABILITY_INIT_CHECKPOINT only to override automatic selection."
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
    if [[ -z "${INIT_CHECKPOINT}" ]]; then
        local preferred="${HOME}/n2_checkpoints/isaac_9050/model_9050.pt"
        if [[ -f "${preferred}" ]]; then
            INIT_CHECKPOINT="${preferred}"
        else
            local matches=()
            local newest
            local checkpoint
            shopt -s nullglob
            matches=(
                "${ROOT_DIR}/logs/n2_stairs_walk/"*"_isaac_l4_guarded_pilot_from_9000_s${TRAIN_SEED}/model_9050.pt"
            )
            shopt -u nullglob
            if [[ "${#matches[@]}" -gt 0 ]]; then
                newest="${matches[0]}"
                for checkpoint in "${matches[@]:1}"; do
                    if [[ "${checkpoint}" -nt "${newest}" ]]; then
                        newest="${checkpoint}"
                    fi
                done
                INIT_CHECKPOINT="${newest}"
            fi
        fi
    fi
    if [[ -z "${INIT_CHECKPOINT}" || ! -f "${INIT_CHECKPOINT}" ]]; then
        echo "Cannot find model_9050.pt; set N2_STABILITY_INIT_CHECKPOINT." >&2
        exit 2
    fi
    INIT_CHECKPOINT="$(readlink -f "${INIT_CHECKPOINT}")"
    checkpoint_iteration "${INIT_CHECKPOINT}" >/dev/null
}

checkpoint_iteration() {
    local name
    name="$(basename "$1")"
    if [[ ! "${name}" =~ ^model_([0-9]+)\.pt$ ]]; then
        echo "Checkpoint must be named model_<iteration>.pt: $1" >&2
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

configure_profile() {
    local profile="$1"
    case "${profile}" in
        conservative)
            PROFILE_LEARNING_RATE="2.5e-7"
            PROFILE_ACTION_NOISE="0.025"
            PROFILE_REFERENCE_COEFF="3.0"
            PROFILE_SYMMETRY_COEFF="0.02"
            PROFILE_OBSERVATION_NOISE="0.02"
            PROFILE_REWARD_OVERRIDES="action_rate=-0.50,action_smoothness=-0.30,dof_acc=-5e-7,stairs_lateral_drift=-18,stairs_heading_alignment=4,stairs_foothold_lateral=1,stairs_foothold_lateral_error=-2,stairs_stride_symmetry=-8"
            ;;
        smooth)
            PROFILE_LEARNING_RATE="5e-7"
            PROFILE_ACTION_NOISE="0.030"
            PROFILE_REFERENCE_COEFF="2.5"
            PROFILE_SYMMETRY_COEFF="0.04"
            PROFILE_OBSERVATION_NOISE="0.03"
            PROFILE_REWARD_OVERRIDES="action_rate=-0.65,action_smoothness=-0.40,dof_acc=-6e-7,stairs_lateral_drift=-18,stairs_heading_alignment=4,stairs_foothold_lateral=1,stairs_foothold_lateral_error=-2,stairs_stride_symmetry=-8"
            ;;
        balance)
            PROFILE_LEARNING_RATE="5e-7"
            PROFILE_ACTION_NOISE="0.030"
            PROFILE_REFERENCE_COEFF="2.5"
            PROFILE_SYMMETRY_COEFF="0.08"
            PROFILE_OBSERVATION_NOISE="0.02"
            PROFILE_REWARD_OVERRIDES="action_rate=-0.50,action_smoothness=-0.30,dof_acc=-5e-7,stairs_lateral_drift=-22,stairs_heading_alignment=5,stairs_foothold_lateral=1.5,stairs_foothold_lateral_error=-3,stairs_stride_symmetry=-14"
            ;;
        *)
            echo "Unknown stability profile: ${profile}" >&2
            return 2
            ;;
    esac
}

train_candidate() {
    local source_checkpoint="$1"
    local profile="$2"
    local round="$3"
    local candidate_index="$4"
    local iterations="$5"
    local source_iteration
    local target_iteration
    local source_run
    local run_name
    local output_run
    local candidate_seed

    configure_profile "${profile}"
    source_iteration="$(checkpoint_iteration "${source_checkpoint}")"
    target_iteration=$((source_iteration + iterations))
    source_run="$(dirname "${source_checkpoint}")"
    candidate_seed=$((TRAIN_SEED + round * 100 + candidate_index))
    run_name="stability_search_r${round}_${profile}_from_${source_iteration}_s${candidate_seed}"

    echo "ISAAC_STABILITY_MICRO_TRAIN round=${round} profile=${profile} iterations=${iterations} checkpoint=${source_checkpoint}"
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
        "--seed=${candidate_seed}" \
        "--terrain_level_mix=${TERRAIN_MIX}" \
        "--command_speed=${COMMAND_SPEED}" \
        --reset_optimizer \
        "--learning_rate=${PROFILE_LEARNING_RATE}" \
        --fixed_learning_rate \
        "--action_noise_std=${PROFILE_ACTION_NOISE}" \
        --freeze_action_noise \
        --actor_trainable_layers=1 \
        "--actor_reference_loss_coeff=${PROFILE_REFERENCE_COEFF}" \
        "--symmetry_loss_coeff=${PROFILE_SYMMETRY_COEFF}" \
        "--observation_noise_level=${PROFILE_OBSERVATION_NOISE}" \
        "--reward_scale_overrides=${PROFILE_REWARD_OVERRIDES}"

    output_run="$(latest_run_directory "${run_name}")"
    CANDIDATE_CHECKPOINT="${output_run}/model_${target_iteration}.pt"
    if [[ ! -f "${CANDIDATE_CHECKPOINT}" ]]; then
        echo "Micro-candidate checkpoint was not saved: ${CANDIDATE_CHECKPOINT}" >&2
        return 1
    fi
}

evaluate_checkpoint() {
    local checkpoint="$1"
    local output="$2"
    local env_count="$3"
    local eval_seed="$4"
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
        "--seed=${eval_seed}" \
        --terrain_levels=0,1,2,3,4 \
        "--command_speed=${COMMAND_SPEED}" \
        --episodes_per_env=1 \
        "--output=${output}"
}

decision_value() {
    local decision_path="$1"
    local expression="$2"
    python -c "import json,sys; d=json.load(open(sys.argv[1])); print(${expression})" \
        "${decision_path}"
}

run_tournament() {
    local baseline_csv="$1"
    local baseline_checkpoint="$2"
    local decision_path="$3"
    local episodes="$4"
    shift 4
    local command=(
        python humanoid/scripts/isaac_stability_tournament.py
        "--baseline=${baseline_csv}"
        "--baseline-checkpoint=${baseline_checkpoint}"
        "--episodes=${episodes}"
        "--output=${decision_path}"
    )
    local candidate
    for candidate in "$@"; do
        command+=(--candidate "${candidate}")
    done
    run_child "${command[@]}"
}

run_search() {
    require_checkpoint
    require_positive_integer N2_STABILITY_MICRO_ITERATIONS "${MICRO_ITERATIONS}"
    require_positive_integer N2_STABILITY_SEARCH_ROUNDS "${SEARCH_ROUNDS}"
    require_positive_integer N2_STABILITY_EVAL_ENVS "${EVAL_ENVS}"
    require_positive_integer N2_STABILITY_HOLDOUT_ENVS "${HOLDOUT_ENVS}"

    trap terminate_driver TERM INT
    trap cleanup_driver EXIT

    local timestamp
    local work_dir
    local initial_checkpoint
    local selected_checkpoint
    local baseline_csv
    local round
    local profile
    local candidate_index
    local candidate_csv
    local decision_json
    local winner_checkpoint
    local winner_evaluation
    local improved
    local selected_iteration
    local selected_copy
    local final_evaluation
    local holdout_seed
    local holdout_baseline_csv
    local holdout_candidate_csv
    local holdout_decision_json
    local profiles=(conservative smooth balance)
    local candidate_specs=()

    timestamp="$(date +%m%d_%H-%M-%S)"
    work_dir="${WORK_ROOT}/${timestamp}"
    mkdir -p "${work_dir}"
    initial_checkpoint="${INIT_CHECKPOINT}"
    selected_checkpoint="${INIT_CHECKPOINT}"
    baseline_csv="${work_dir}/round_0_baseline.csv"
    improved="False"

    echo "ISAAC_STABILITY_SEARCH_START checkpoint=${selected_checkpoint}"
    echo "ISAAC_STABILITY_SEARCH_PLAN rounds=${SEARCH_ROUNDS} micro_iterations=${MICRO_ITERATIONS} profiles=conservative,smooth,balance terrain_mix=${TERRAIN_MIX}"
    evaluate_checkpoint \
        "${selected_checkpoint}" "${baseline_csv}" "${EVAL_ENVS}" "${TRAIN_SEED}"

    for ((round = 1; round <= SEARCH_ROUNDS; round++)); do
        candidate_specs=()
        candidate_index=0
        for profile in "${profiles[@]}"; do
            candidate_index=$((candidate_index + 1))
            train_candidate \
                "${selected_checkpoint}" "${profile}" "${round}" \
                "${candidate_index}" "${MICRO_ITERATIONS}"
            candidate_csv="${work_dir}/round_${round}_${profile}.csv"
            evaluate_checkpoint \
                "${CANDIDATE_CHECKPOINT}" "${candidate_csv}" \
                "${EVAL_ENVS}" "${TRAIN_SEED}"
            candidate_specs+=(
                "${profile}|${candidate_csv}|${CANDIDATE_CHECKPOINT}"
            )
        done

        decision_json="${work_dir}/round_${round}_decision.json"
        run_tournament \
            "${baseline_csv}" "${selected_checkpoint}" "${decision_json}" \
            "${EVAL_ENVS}" "${candidate_specs[@]}"
        winner_checkpoint="$(
            decision_value "${decision_json}" 'd["winner"]["checkpoint"]'
        )"
        winner_evaluation="$(
            decision_value "${decision_json}" 'd["winner"]["evaluation"]'
        )"
        improved="$(
            decision_value "${decision_json}" 'd["improved"]'
        )"
        if [[ "${improved}" != "True" ]]; then
            echo "ISAAC_STABILITY_SEARCH_STOP round=${round} reason=no_safe_style_improvement"
            break
        fi
        selected_checkpoint="${winner_checkpoint}"
        baseline_csv="${winner_evaluation}"
        echo "ISAAC_STABILITY_SEARCH_ADVANCE round=${round} checkpoint=${selected_checkpoint}"
    done

    final_evaluation="${baseline_csv}"
    if [[ "${selected_checkpoint}" != "${initial_checkpoint}" ]]; then
        holdout_seed=$((TRAIN_SEED + 1000))
        holdout_baseline_csv="${work_dir}/holdout_baseline.csv"
        holdout_candidate_csv="${work_dir}/holdout_candidate.csv"
        holdout_decision_json="${work_dir}/holdout_decision.json"
        echo "ISAAC_STABILITY_HOLDOUT seed=${holdout_seed} envs=${HOLDOUT_ENVS}"
        evaluate_checkpoint \
            "${initial_checkpoint}" "${holdout_baseline_csv}" \
            "${HOLDOUT_ENVS}" "${holdout_seed}"
        evaluate_checkpoint \
            "${selected_checkpoint}" "${holdout_candidate_csv}" \
            "${HOLDOUT_ENVS}" "${holdout_seed}"
        run_tournament \
            "${holdout_baseline_csv}" "${initial_checkpoint}" \
            "${holdout_decision_json}" "${HOLDOUT_ENVS}" \
            "selected|${holdout_candidate_csv}|${selected_checkpoint}"
        selected_checkpoint="$(
            decision_value "${holdout_decision_json}" \
                'd["winner"]["checkpoint"]'
        )"
        final_evaluation="$(
            decision_value "${holdout_decision_json}" \
                'd["winner"]["evaluation"]'
        )"
        improved="$(
            decision_value "${holdout_decision_json}" 'd["improved"]'
        )"
        if [[ "${improved}" == "True" ]]; then
            echo "ISAAC_STABILITY_HOLDOUT_ACCEPT checkpoint=${selected_checkpoint}"
        else
            echo "ISAAC_STABILITY_HOLDOUT_REJECT restoring=${initial_checkpoint}"
        fi
    fi

    selected_iteration="$(checkpoint_iteration "${selected_checkpoint}")"
    selected_copy="${RESULT_DIR}/model_${selected_iteration}.pt"
    cp -f "${selected_checkpoint}" "${selected_copy}"
    cp -f "${final_evaluation}" "${RESULT_DIR}/evaluation_all_levels.csv"
    printf '%s\n' "${selected_copy}" > "${RESULT_DIR}/selected_checkpoint.txt"
    printf '%s\n' \
        "improved=${improved}" \
        "source=${initial_checkpoint}" \
        "selected=${selected_copy}" \
        "evaluation=${RESULT_DIR}/evaluation_all_levels.csv" \
        > "${RESULT_DIR}/search_summary.txt"

    echo "N2_ISAAC_STABILITY_IMPROVED=${improved}"
    echo "N2_ISAAC_STABILITY_CHECKPOINT=${selected_copy}"
    echo "N2_ISAAC_STABILITY_EVALUATION=${RESULT_DIR}/evaluation_all_levels.csv"
}

smoke_train() {
    require_checkpoint
    TRAIN_ENVS="${N2_NUM_ENVS:-32}"
    train_candidate "${INIT_CHECKPOINT}" conservative 0 1 2
    echo "N2_ISAAC_STABILITY_SMOKE_CHECKPOINT=${CANDIDATE_CHECKPOINT}"
}

launch_search() {
    local label="$1"
    shift
    if active_pid >/dev/null; then
        echo "Isaac stability search is already running PID=$(active_pid)." >&2
        exit 2
    fi
    local timestamp
    local log_path
    local pid
    timestamp="$(date +%m%d_%H-%M-%S)"
    log_path="${LAUNCHER_DIR}/isaac_stability_search_${label}_s${TRAIN_SEED}_${timestamp}.log"
    printf '%s\n' "${log_path}" > "${ACTIVE_LOG_FILE}"
    nohup env "$@" bash "$0" _run > "${log_path}" 2>&1 &
    pid=$!
    printf '%s\n' "${pid}" > "${PID_FILE}"
    echo "Started Isaac stability ${label} PID=${pid}"
    echo "Log: ${log_path}"
    echo "Watch: $0 log"
}

case "${MODE}" in
    smoke)
        smoke_train
        ;;
    pilot)
        launch_search pilot \
            N2_STABILITY_MICRO_ITERATIONS=4 \
            N2_STABILITY_SEARCH_ROUNDS=1 \
            N2_STABILITY_EVAL_ENVS=64 \
            N2_STABILITY_HOLDOUT_ENVS=128
        ;;
    long)
        launch_search long
        ;;
    _run)
        run_search
        ;;
    status)
        if pid="$(active_pid)"; then
            echo "Isaac stability search is running PID=${pid}"
        else
            echo "No Isaac stability search is running."
        fi
        if [[ -f "${ACTIVE_LOG_FILE}" ]]; then
            log_path="$(<"${ACTIVE_LOG_FILE}")"
            echo "Latest log: ${log_path}"
            [[ -f "${log_path}" ]] && tail -n 40 "${log_path}"
        fi
        ;;
    log)
        if [[ ! -f "${ACTIVE_LOG_FILE}" ]]; then
            echo "No stability search log has been recorded." >&2
            exit 2
        fi
        log_path="$(<"${ACTIVE_LOG_FILE}")"
        echo "Following ${log_path}"
        tail -f "${log_path}"
        ;;
    stop)
        if pid="$(active_pid)"; then
            kill -TERM "${pid}"
            echo "Sent SIGTERM to stability search PID=${pid}."
        else
            echo "No Isaac stability search is running."
        fi
        ;;
    *)
        usage
        exit 2
        ;;
esac
