#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-}"
TRAIN_SEED="${N2_SEED:-42}"
TRAIN_DEVICE="${N2_DEVICE:-cuda:0}"
TRAIN_ENVS="${N2_NUM_ENVS:-256}"
INIT_CHECKPOINT="${N2_STABILITY_INIT_CHECKPOINT:-}"

# This is one continuous adaptation run.  Periodic checkpoints are evaluated
# afterwards; training is not restarted from model_9050 for each candidate.
TRAIN_ITERATIONS="${N2_STABILITY_TRAIN_ITERATIONS:-250}"
CHECKPOINT_INTERVAL="${N2_STABILITY_CHECKPOINT_INTERVAL:-25}"
EVAL_ENVS="${N2_STABILITY_EVAL_ENVS:-128}"
HOLDOUT_ENVS="${N2_STABILITY_HOLDOUT_ENVS:-256}"
TERRAIN_MIX="${N2_STABILITY_TERRAIN_MIX:-0,1,2,3,4,4,4,4}"
COMMAND_SPEED="${N2_STABILITY_COMMAND_SPEED:-0.18}"

# model_9050 already climbs.  Give the last two Actor layers enough freedom to
# reshape the gait, while a moderate teacher anchor and conservative learning
# rate protect the climbing skill.  The previous 4-iteration search used a
# 5--10x smaller rate, a 25x stronger anchor, and only the output layer.
LEARNING_RATE="${N2_STABILITY_LEARNING_RATE:-2.0e-6}"
ACTION_NOISE="${N2_STABILITY_ACTION_NOISE:-0.08}"
REFERENCE_COEFF="${N2_STABILITY_REFERENCE_COEFF:-0.10}"
SYMMETRY_COEFF="${N2_STABILITY_SYMMETRY_COEFF:-0.002}"
ACTOR_LAYERS="${N2_STABILITY_ACTOR_LAYERS:-2}"
OBSERVATION_NOISE="${N2_STABILITY_OBSERVATION_NOISE:-0.10}"
REWARD_OVERRIDES="${N2_STABILITY_REWARD_OVERRIDES:-action_rate=-0.18,action_smoothness=-0.08,dof_acc=-3e-7,stairs_lateral_drift=-16,stairs_heading_alignment=4,stairs_stride_symmetry=-4,stairs_alternating_tread=2,stairs_repeated_lead=-2,stairs_same_tread_join=-2}"

LAUNCHER_DIR="${ROOT_DIR}/logs/isaac_launcher"
TRAIN_ROOT="${ROOT_DIR}/logs/n2_stairs_stability"
PID_FILE="${LAUNCHER_DIR}/n2_stability_continuous_s${TRAIN_SEED}.pid"
ACTIVE_LOG_FILE="${LAUNCHER_DIR}/n2_stability_continuous_s${TRAIN_SEED}.logpath"
WORK_ROOT="${LAUNCHER_DIR}/stability_continuous_s${TRAIN_SEED}"
RESULT_DIR="${LAUNCHER_DIR}/stability_selected_s${TRAIN_SEED}"

mkdir -p "${LAUNCHER_DIR}" "${WORK_ROOT}" "${RESULT_DIR}"

CHILD_PID=""
TRAINED_RUN=""
TARGET_ITERATION=""

usage() {
    echo "Usage: $0 smoke|pilot|long|status|log|stop"
    echo "pilot: one 75-iteration run; long: one 250-iteration run."
    echo "The guarded model_9050.pt is selected automatically."
    echo "Set N2_STABILITY_INIT_CHECKPOINT only to override it."
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, received ${value}" >&2
        exit 2
    fi
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

train_trajectory() {
    local source_checkpoint="$1"
    local extra_iterations="$2"
    local checkpoint_interval="$3"
    local source_iteration
    local source_run
    local run_name

    source_iteration="$(checkpoint_iteration "${source_checkpoint}")"
    TARGET_ITERATION=$((source_iteration + extra_iterations))
    source_run="$(dirname "${source_checkpoint}")"
    run_name="stability_continuous_from_${source_iteration}_to_${TARGET_ITERATION}_s${TRAIN_SEED}"

    echo "ISAAC_STABILITY_CONTINUOUS_TRAIN source=${source_iteration} target=${TARGET_ITERATION} iterations=${extra_iterations}"
    run_child python -u humanoid/scripts/train.py \
        --task=n2_stairs_walk \
        --resume \
        "--load_run=${source_run}" \
        "--checkpoint=${source_iteration}" \
        --headless \
        "--sim_device=${TRAIN_DEVICE}" \
        "--rl_device=${TRAIN_DEVICE}" \
        "--num_envs=${TRAIN_ENVS}" \
        "--max_iterations=${TARGET_ITERATION}" \
        --experiment_name=n2_stairs_stability \
        "--run_name=${run_name}" \
        "--seed=${TRAIN_SEED}" \
        "--terrain_level_mix=${TERRAIN_MIX}" \
        "--command_speed=${COMMAND_SPEED}" \
        --reset_optimizer \
        "--learning_rate=${LEARNING_RATE}" \
        --fixed_learning_rate \
        "--action_noise_std=${ACTION_NOISE}" \
        --freeze_action_noise \
        "--actor_trainable_layers=${ACTOR_LAYERS}" \
        "--actor_reference_loss_coeff=${REFERENCE_COEFF}" \
        "--symmetry_loss_coeff=${SYMMETRY_COEFF}" \
        "--observation_noise_level=${OBSERVATION_NOISE}" \
        "--reward_scale_overrides=${REWARD_OVERRIDES}" \
        "--save_interval=${checkpoint_interval}"

    TRAINED_RUN="$(latest_run_directory "${run_name}")"
    if [[ ! -f "${TRAINED_RUN}/model_${TARGET_ITERATION}.pt" ]]; then
        echo "Final continuous checkpoint was not saved: ${TRAINED_RUN}/model_${TARGET_ITERATION}.pt" >&2
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

run_continuous() {
    require_checkpoint
    require_positive_integer N2_STABILITY_TRAIN_ITERATIONS "${TRAIN_ITERATIONS}"
    require_positive_integer N2_STABILITY_CHECKPOINT_INTERVAL "${CHECKPOINT_INTERVAL}"
    require_positive_integer N2_STABILITY_EVAL_ENVS "${EVAL_ENVS}"
    require_positive_integer N2_STABILITY_HOLDOUT_ENVS "${HOLDOUT_ENVS}"
    require_positive_integer N2_STABILITY_ACTOR_LAYERS "${ACTOR_LAYERS}"

    trap terminate_driver TERM INT
    trap cleanup_driver EXIT

    local timestamp
    local work_dir
    local baseline_csv
    local source_iteration
    local checkpoint
    local iteration
    local candidate_csv
    local decision_json
    local selected_checkpoint
    local final_evaluation
    local improved
    local holdout_seed
    local holdout_baseline_csv
    local holdout_candidate_csv
    local holdout_decision_json
    local selected_iteration
    local selected_copy
    local candidate_count
    local expected_transitions
    local optimizer_updates
    local candidate_specs=()

    timestamp="$(date +%m%d_%H-%M-%S)"
    work_dir="${WORK_ROOT}/${timestamp}"
    mkdir -p "${work_dir}"
    source_iteration="$(checkpoint_iteration "${INIT_CHECKPOINT}")"
    baseline_csv="${work_dir}/baseline.csv"
    expected_transitions=$((TRAIN_ITERATIONS * TRAIN_ENVS * 24))
    optimizer_updates=$((TRAIN_ITERATIONS * 5 * 4))

    echo "ISAAC_STABILITY_CONTINUOUS_START checkpoint=${INIT_CHECKPOINT}"
    echo "ISAAC_STABILITY_CONTINUOUS_PLAN train_iterations=${TRAIN_ITERATIONS} checkpoint_interval=${CHECKPOINT_INTERVAL} actor_layers=${ACTOR_LAYERS} learning_rate=${LEARNING_RATE} reference=${REFERENCE_COEFF} symmetry=${SYMMETRY_COEFF} noise=${ACTION_NOISE} terrain_mix=${TERRAIN_MIX}"
    echo "ISAAC_STABILITY_TRAINING_VOLUME transitions=${expected_transitions} optimizer_minibatch_updates=${optimizer_updates}"
    evaluate_checkpoint \
        "${INIT_CHECKPOINT}" "${baseline_csv}" "${EVAL_ENVS}" "${TRAIN_SEED}"

    train_trajectory \
        "${INIT_CHECKPOINT}" "${TRAIN_ITERATIONS}" "${CHECKPOINT_INTERVAL}"

    candidate_count=0
    while IFS= read -r checkpoint; do
        iteration="$(checkpoint_iteration "${checkpoint}")"
        if (( iteration <= source_iteration || iteration > TARGET_ITERATION )); then
            continue
        fi
        if (( iteration != TARGET_ITERATION && iteration % CHECKPOINT_INTERVAL != 0 )); then
            continue
        fi
        candidate_count=$((candidate_count + 1))
        candidate_csv="${work_dir}/checkpoint_${iteration}.csv"
        echo "ISAAC_STABILITY_SCREEN iteration=${iteration} candidate=${candidate_count}"
        evaluate_checkpoint \
            "${checkpoint}" "${candidate_csv}" "${EVAL_ENVS}" "${TRAIN_SEED}"
        candidate_specs+=(
            "iter_${iteration}|${candidate_csv}|${checkpoint}"
        )
    done < <(
        find "${TRAINED_RUN}" -maxdepth 1 -type f -name 'model_*.pt' \
            | sort -V
    )

    if (( candidate_count == 0 )); then
        echo "No periodic checkpoints were found in ${TRAINED_RUN}" >&2
        exit 1
    fi

    decision_json="${work_dir}/trajectory_decision.json"
    run_tournament \
        "${baseline_csv}" "${INIT_CHECKPOINT}" "${decision_json}" \
        "${EVAL_ENVS}" "${candidate_specs[@]}"
    selected_checkpoint="$(
        decision_value "${decision_json}" 'd["winner"]["checkpoint"]'
    )"
    final_evaluation="$(
        decision_value "${decision_json}" 'd["winner"]["evaluation"]'
    )"
    improved="$(
        decision_value "${decision_json}" 'd["improved"]'
    )"

    if [[ "${selected_checkpoint}" != "${INIT_CHECKPOINT}" ]]; then
        holdout_seed=$((TRAIN_SEED + 1000))
        holdout_baseline_csv="${work_dir}/holdout_baseline.csv"
        holdout_candidate_csv="${work_dir}/holdout_candidate.csv"
        holdout_decision_json="${work_dir}/holdout_decision.json"
        echo "ISAAC_STABILITY_HOLDOUT seed=${holdout_seed} envs=${HOLDOUT_ENVS}"
        evaluate_checkpoint \
            "${INIT_CHECKPOINT}" "${holdout_baseline_csv}" \
            "${HOLDOUT_ENVS}" "${holdout_seed}"
        evaluate_checkpoint \
            "${selected_checkpoint}" "${holdout_candidate_csv}" \
            "${HOLDOUT_ENVS}" "${holdout_seed}"
        run_tournament \
            "${holdout_baseline_csv}" "${INIT_CHECKPOINT}" \
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
            echo "ISAAC_STABILITY_HOLDOUT_REJECT restoring=${INIT_CHECKPOINT}"
        fi
    fi

    selected_iteration="$(checkpoint_iteration "${selected_checkpoint}")"
    selected_copy="${RESULT_DIR}/model_${selected_iteration}.pt"
    cp -f "${selected_checkpoint}" "${selected_copy}"
    cp -f "${final_evaluation}" "${RESULT_DIR}/evaluation_all_levels.csv"
    printf '%s\n' "${selected_copy}" > "${RESULT_DIR}/selected_checkpoint.txt"
    printf '%s\n' \
        "improved=${improved}" \
        "source=${INIT_CHECKPOINT}" \
        "selected=${selected_copy}" \
        "evaluation=${RESULT_DIR}/evaluation_all_levels.csv" \
        "trained_run=${TRAINED_RUN}" \
        "screened_checkpoints=${candidate_count}" \
        > "${RESULT_DIR}/search_summary.txt"

    echo "N2_ISAAC_STABILITY_IMPROVED=${improved}"
    echo "N2_ISAAC_STABILITY_CHECKPOINT=${selected_copy}"
    echo "N2_ISAAC_STABILITY_EVALUATION=${RESULT_DIR}/evaluation_all_levels.csv"
    echo "N2_ISAAC_STABILITY_TRAINED_RUN=${TRAINED_RUN}"
}

smoke_train() {
    require_checkpoint
    TRAIN_ENVS="${N2_NUM_ENVS:-32}"
    train_trajectory "${INIT_CHECKPOINT}" 2 1
    echo "N2_ISAAC_STABILITY_SMOKE_CHECKPOINT=${TRAINED_RUN}/model_${TARGET_ITERATION}.pt"
}

launch_continuous() {
    local label="$1"
    shift
    if active_pid >/dev/null; then
        echo "Isaac stability training is already running PID=$(active_pid)." >&2
        exit 2
    fi
    local timestamp
    local log_path
    local pid
    timestamp="$(date +%m%d_%H-%M-%S)"
    log_path="${LAUNCHER_DIR}/isaac_stability_continuous_${label}_s${TRAIN_SEED}_${timestamp}.log"
    printf '%s\n' "${log_path}" > "${ACTIVE_LOG_FILE}"
    nohup env "$@" bash "$0" _run > "${log_path}" 2>&1 &
    pid=$!
    printf '%s\n' "${pid}" > "${PID_FILE}"
    echo "Started Isaac stability ${label} PID=${pid}"
    echo "Log: ${log_path}"
    echo "Watch: bash humanoid/scripts/run_isaac_stability_curriculum.sh log"
}

case "${MODE}" in
    smoke)
        smoke_train
        ;;
    pilot)
        launch_continuous pilot \
            N2_STABILITY_TRAIN_ITERATIONS=75 \
            N2_STABILITY_CHECKPOINT_INTERVAL=15 \
            N2_STABILITY_EVAL_ENVS=64 \
            N2_STABILITY_HOLDOUT_ENVS=128
        ;;
    long)
        launch_continuous long
        ;;
    _run)
        run_continuous
        ;;
    status)
        if pid="$(active_pid)"; then
            echo "Isaac stability training is running PID=${pid}"
        else
            echo "No Isaac stability training is running."
        fi
        if [[ -f "${ACTIVE_LOG_FILE}" ]]; then
            log_path="$(<"${ACTIVE_LOG_FILE}")"
            echo "Latest log: ${log_path}"
            [[ -f "${log_path}" ]] && tail -n 40 "${log_path}"
        fi
        ;;
    log)
        if [[ ! -f "${ACTIVE_LOG_FILE}" ]]; then
            echo "No stability training log has been recorded." >&2
            exit 2
        fi
        log_path="$(<"${ACTIVE_LOG_FILE}")"
        echo "Following ${log_path}"
        tail -f "${log_path}"
        ;;
    stop)
        if pid="$(active_pid)"; then
            kill -TERM "${pid}"
            echo "Sent SIGTERM to stability training PID=${pid}."
        else
            echo "No Isaac stability training is running."
        fi
        ;;
    *)
        usage
        exit 2
        ;;
esac
