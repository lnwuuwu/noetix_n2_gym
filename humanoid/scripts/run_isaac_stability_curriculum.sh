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
TRAIN_ITERATIONS="${N2_STABILITY_TRAIN_ITERATIONS:-400}"
CHECKPOINT_INTERVAL="${N2_STABILITY_CHECKPOINT_INTERVAL:-20}"
EVAL_ENVS="${N2_STABILITY_EVAL_ENVS:-128}"
HOLDOUT_ENVS="${N2_STABILITY_HOLDOUT_ENVS:-256}"
# The inherited policy learns a low-step shuffle on 2/4 cm stairs. Focus
# adaptation on 6--10 cm while retaining three riser heights.
TERRAIN_MIX="${N2_STABILITY_TERRAIN_MIX:-2,3,4,4,4,4,4,4}"
COMMAND_SPEED="${N2_STABILITY_COMMAND_SPEED:-0.18}"

# model_9050 already climbs.  Give the last two Actor layers enough freedom to
# reshape the gait, while a moderate teacher anchor and conservative learning
# rate protect the climbing skill.  The previous 4-iteration search used a
# 5--10x smaller rate, a 25x stronger anchor, and only the output layer.
LEARNING_RATE="${N2_STABILITY_LEARNING_RATE:-1.0e-6}"
ACTION_NOISE="${N2_STABILITY_ACTION_NOISE:-0.05}"
REFERENCE_COEFF="${N2_STABILITY_REFERENCE_COEFF:-0.20}"
SYMMETRIZE_REFERENCE="${N2_STABILITY_SYMMETRIZE_REFERENCE:-False}"
REFERENCE_MIRROR_BLEND="${N2_STABILITY_REFERENCE_MIRROR_BLEND:-0.5}"
SYMMETRY_COEFF="${N2_STABILITY_SYMMETRY_COEFF:-0.006}"
POLICY_LOSS_SCALE="${N2_STABILITY_POLICY_LOSS_SCALE:-1.0}"
ACTOR_LAYERS="${N2_STABILITY_ACTOR_LAYERS:-2}"
OBSERVATION_NOISE="${N2_STABILITY_OBSERVATION_NOISE:-0.05}"
REWARD_OVERRIDES="${N2_STABILITY_REWARD_OVERRIDES:-action_rate=-0.16,action_smoothness=-0.12,dof_acc=-4e-7,stairs_lateral_drift=-18,stairs_heading_alignment=4,stairs_stride_symmetry=-10,stairs_foothold_lateral=1.5,stairs_foothold_lateral_error=-4,stairs_foot_crossover=-10,stairs_foot_lane_error=-6,stairs_single_support_stability=-5,stairs_right_support_stability=-4,stairs_alternating_tread=2,stairs_repeated_lead=-2,stairs_same_tread_join=-2}"
VIEW_PORT="${N2_STREAM_PORT:-18080}"
SOURCE_APPROVED="${N2_STABILITY_SOURCE_APPROVED:-False}"
CORRECTION_PREFLIGHT="${N2_STABILITY_CORRECTION_PREFLIGHT:-False}"
SELECTION_MODE="${N2_ISAAC_STABILITY_SELECTION_MODE:-balanced}"
export N2_ISAAC_STABILITY_SELECTION_MODE="${SELECTION_MODE}"

LAUNCHER_DIR="${ROOT_DIR}/logs/isaac_launcher"
TRAIN_ROOT="${ROOT_DIR}/logs/n2_stairs_stability"
PID_FILE="${LAUNCHER_DIR}/n2_stability_continuous_s${TRAIN_SEED}.pid"
ACTIVE_LOG_FILE="${LAUNCHER_DIR}/n2_stability_continuous_s${TRAIN_SEED}.logpath"
WORK_ROOT="${LAUNCHER_DIR}/stability_continuous_s${TRAIN_SEED}"
RESULT_DIR="${LAUNCHER_DIR}/stability_selected_s${TRAIN_SEED}"
SELECTED_BLEND_FILE="${RESULT_DIR}/selected_policy_blend.txt"

mkdir -p "${LAUNCHER_DIR}" "${WORK_ROOT}" "${RESULT_DIR}"

CHILD_PID=""
TRAINED_RUN=""
TARGET_ITERATION=""
PREFLIGHT_EVALUATION=""

usage() {
    echo "Usage: $0 smoke|pilot|long|diagnose|correct|final|status|log|stop|view"
    echo "pilot: one 80-iteration run; long: one 400-iteration run."
    echo "diagnose: zero-training mirrored-policy safety/style preflight."
    echo "correct/final: preflight-gated 100-iteration full-Actor distillation."
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
    if [[ "${name}" =~ ^model_([0-9]+)\.pt$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ "${name}" == "model_best.pt" ]]; then
        python -c \
            'import sys,torch; print(int(torch.load(sys.argv[1], map_location="cpu")["iter"]))' \
            "$1"
        return $?
    fi
    echo "Checkpoint must be model_<iteration>.pt or model_best.pt: $1" >&2
    return 1
}

require_checkpoint() {
    local checkpoint_iter
    local numeric_checkpoint
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
    checkpoint_iter="$(checkpoint_iteration "${INIT_CHECKPOINT}")"
    if [[ "$(basename "${INIT_CHECKPOINT}")" == "model_best.pt" ]]; then
        numeric_checkpoint="$(
            dirname "${INIT_CHECKPOINT}"
        )/model_${checkpoint_iter}.pt"
        if [[ ! -f "${numeric_checkpoint}" ]]; then
            cp -f "${INIT_CHECKPOINT}" "${numeric_checkpoint}"
            echo "Materialized numbered checkpoint: ${numeric_checkpoint}"
        fi
        INIT_CHECKPOINT="${numeric_checkpoint}"
    fi
}

require_approved_selection() {
    local summary_path="${RESULT_DIR}/search_summary.txt"
    local selected_path="${RESULT_DIR}/selected_checkpoint.txt"
    if [[ ! -f "${summary_path}" || ! -f "${selected_path}" ]]; then
        echo "No completed guarded selection exists yet." >&2
        echo "Wait for the current holdout to finish before using final/view." >&2
        exit 2
    fi
    if ! grep -qx 'improved=True' "${summary_path}" \
        && ! grep -qx 'approved=True' "${summary_path}"; then
        echo "The independent holdout did not approve a new policy." >&2
        echo "Final refinement is blocked to avoid spending GPU time on an unverified checkpoint." >&2
        exit 2
    fi
    INIT_CHECKPOINT="$(<"${selected_path}")"
    if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
        echo "Selected checkpoint does not exist: ${INIT_CHECKPOINT}" >&2
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
    local reference_options=()

    source_iteration="$(checkpoint_iteration "${source_checkpoint}")"
    TARGET_ITERATION=$((source_iteration + extra_iterations))
    source_run="$(dirname "${source_checkpoint}")"
    run_name="stability_continuous_from_${source_iteration}_to_${TARGET_ITERATION}_s${TRAIN_SEED}"
    if [[ "${SYMMETRIZE_REFERENCE}" == "True" ]]; then
        reference_options+=(--symmetrize_actor_reference)
    elif [[ "${SYMMETRIZE_REFERENCE}" != "False" ]]; then
        echo "N2_STABILITY_SYMMETRIZE_REFERENCE must be True or False." >&2
        return 2
    fi

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
        "${reference_options[@]}" \
        "--actor_reference_mirror_blend=${REFERENCE_MIRROR_BLEND}" \
        "--actor_policy_loss_scale=${POLICY_LOSS_SCALE}" \
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
    local reflection_blend="${5:-0.0}"
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
        "--policy_symmetry_blend=${reflection_blend}" \
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

run_reflection_preflight() {
    # Select a safe mirrored-policy blend before spending time training.
    local timestamp
    local work_dir
    local baseline_csv
    local blend
    local candidate_csv
    local decision_json
    local winner_name
    local selected_blend
    local candidate_specs=()

    timestamp="$(date +%m%d_%H-%M-%S)"
    work_dir="${WORK_ROOT}/reflection_preflight_${timestamp}"
    mkdir -p "${work_dir}"
    baseline_csv="${work_dir}/blend_0.00.csv"
    echo "ISAAC_SYMMETRY_PREFLIGHT checkpoint=${INIT_CHECKPOINT}"
    evaluate_checkpoint \
        "${INIT_CHECKPOINT}" "${baseline_csv}" "${EVAL_ENVS}" \
        "${TRAIN_SEED}" 0.0
    for blend in 0.05 0.10 0.20 0.35 0.50; do
        candidate_csv="${work_dir}/blend_${blend}.csv"
        echo "ISAAC_SYMMETRY_BLEND_SCREEN blend=${blend}"
        evaluate_checkpoint \
            "${INIT_CHECKPOINT}" "${candidate_csv}" "${EVAL_ENVS}" \
            "${TRAIN_SEED}" "${blend}"
        candidate_specs+=(
            "blend_${blend}|${candidate_csv}|${INIT_CHECKPOINT}"
        )
    done
    decision_json="${work_dir}/reflection_blend_decision.json"
    run_tournament \
        "${baseline_csv}" "${INIT_CHECKPOINT}" "${decision_json}" \
        "${EVAL_ENVS}" "${candidate_specs[@]}"
    winner_name="$(
        decision_value "${decision_json}" 'd["winner"]["name"]'
    )"
    if [[ "${winner_name}" == "baseline" ]]; then
        selected_blend="0.0"
    elif [[ "${winner_name}" == blend_* ]]; then
        selected_blend="${winner_name#blend_}"
    else
        echo "Unexpected reflection preflight winner: ${winner_name}" >&2
        return 1
    fi
    PREFLIGHT_EVALUATION="$(
        decision_value "${decision_json}" 'd["winner"]["evaluation"]'
    )"
    printf '%s\n' "${selected_blend}" > "${SELECTED_BLEND_FILE}"
    cp -f "${decision_json}" "${RESULT_DIR}/reflection_blend_decision.json"
    echo "N2_ISAAC_SYMMETRY_BLEND=${selected_blend}"
    echo "N2_ISAAC_SYMMETRY_PREFLIGHT=${decision_json}"
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

    if [[ "${CORRECTION_PREFLIGHT}" == "True" ]]; then
        run_reflection_preflight
        REFERENCE_MIRROR_BLEND="$(<"${SELECTED_BLEND_FILE}")"
        if [[ "${REFERENCE_MIRROR_BLEND}" == "0.0" ]]; then
            echo "ISAAC_STABILITY_CORRECTION_ABORT no safe mirrored-policy correction passed the deterministic gate."
            echo "No training was started; the approved checkpoint is unchanged."
            return 0
        fi
        echo "ISAAC_STABILITY_DISTILL_TARGET mirror_blend=${REFERENCE_MIRROR_BLEND}"
    elif [[ "${CORRECTION_PREFLIGHT}" != "False" ]]; then
        echo "N2_STABILITY_CORRECTION_PREFLIGHT must be True or False." >&2
        return 2
    fi

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
    local approved
    local holdout_seed
    local holdout_baseline_csv
    local holdout_candidate_csv
    local holdout_decision_json
    local selected_iteration
    local selected_copy
    local best_copy
    local trained_best_copy
    local candidate_count
    local expected_transitions
    local optimizer_updates
    local effective_policy_blend="0.0"
    local candidate_specs=()

    timestamp="$(date +%m%d_%H-%M-%S)"
    work_dir="${WORK_ROOT}/${timestamp}"
    mkdir -p "${work_dir}"
    source_iteration="$(checkpoint_iteration "${INIT_CHECKPOINT}")"
    baseline_csv="${work_dir}/baseline.csv"
    expected_transitions=$((TRAIN_ITERATIONS * TRAIN_ENVS * 24))
    optimizer_updates=$((TRAIN_ITERATIONS * 5 * 4))

    echo "ISAAC_STABILITY_CONTINUOUS_START checkpoint=${INIT_CHECKPOINT}"
    echo "ISAAC_STABILITY_CONTINUOUS_PLAN train_iterations=${TRAIN_ITERATIONS} checkpoint_interval=${CHECKPOINT_INTERVAL} actor_layers=${ACTOR_LAYERS} learning_rate=${LEARNING_RATE} policy_loss_scale=${POLICY_LOSS_SCALE} reference=${REFERENCE_COEFF} symmetric_teacher=${SYMMETRIZE_REFERENCE} mirror_blend=${REFERENCE_MIRROR_BLEND} symmetry=${SYMMETRY_COEFF} noise=${ACTION_NOISE} terrain_mix=${TERRAIN_MIX} selection=${SELECTION_MODE}"
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
    approved="${SOURCE_APPROVED}"

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
            approved="True"
        else
            echo "ISAAC_STABILITY_HOLDOUT_REJECT restoring=${INIT_CHECKPOINT}"
        fi
    fi
    if [[ "${improved}" == "True" ]]; then
        approved="True"
        printf '%s\n' "0.0" > "${SELECTED_BLEND_FILE}"
    elif [[ "${CORRECTION_PREFLIGHT}" == "True" ]]; then
        effective_policy_blend="$(<"${SELECTED_BLEND_FILE}")"
        if [[ -n "${PREFLIGHT_EVALUATION}" ]]; then
            final_evaluation="${PREFLIGHT_EVALUATION}"
        fi
    else
        printf '%s\n' "0.0" > "${SELECTED_BLEND_FILE}"
    fi

    selected_iteration="$(checkpoint_iteration "${selected_checkpoint}")"
    selected_copy="${RESULT_DIR}/model_${selected_iteration}.pt"
    if [[ "${selected_checkpoint}" != "${selected_copy}" ]]; then
        cp -f "${selected_checkpoint}" "${selected_copy}"
    fi
    best_copy="${RESULT_DIR}/model_best.pt"
    trained_best_copy="${TRAINED_RUN}/model_best.pt"
    cp -f "${selected_checkpoint}" "${best_copy}"
    cp -f "${selected_checkpoint}" "${trained_best_copy}"
    cp -f "${final_evaluation}" "${RESULT_DIR}/evaluation_all_levels.csv"
    printf '%s\n' "${selected_copy}" > "${RESULT_DIR}/selected_checkpoint.txt"
    printf '%s\n' \
        "improved=${improved}" \
        "approved=${approved}" \
        "source=${INIT_CHECKPOINT}" \
        "selected=${selected_copy}" \
        "best=${best_copy}" \
        "evaluation=${RESULT_DIR}/evaluation_all_levels.csv" \
        "trained_run=${TRAINED_RUN}" \
        "screened_checkpoints=${candidate_count}" \
        "policy_symmetry_blend=${effective_policy_blend}" \
        > "${RESULT_DIR}/search_summary.txt"

    echo "N2_ISAAC_STABILITY_IMPROVED=${improved}"
    echo "N2_ISAAC_STABILITY_APPROVED=${approved}"
    echo "N2_ISAAC_STABILITY_CHECKPOINT=${selected_copy}"
    echo "N2_ISAAC_STABILITY_BEST=${best_copy}"
    echo "N2_ISAAC_STABILITY_EVALUATION=${RESULT_DIR}/evaluation_all_levels.csv"
    echo "N2_ISAAC_STABILITY_TRAINED_RUN=${TRAINED_RUN}"
    echo "N2_ISAAC_POLICY_SYMMETRY_BLEND=${effective_policy_blend}"
}

smoke_train() {
    require_checkpoint
    TRAIN_ENVS="${N2_NUM_ENVS:-32}"
    train_trajectory "${INIT_CHECKPOINT}" 2 1
    echo "N2_ISAAC_STABILITY_SMOKE_CHECKPOINT=${TRAINED_RUN}/model_${TARGET_ITERATION}.pt"
}

view_selected() {
    require_approved_selection
    if active_pid >/dev/null; then
        echo "Wait for stability training/evaluation to finish before streaming." >&2
        exit 2
    fi
    echo "Selected checkpoint: ${INIT_CHECKPOINT}"
    local policy_blend="0.0"
    if [[ -f "${SELECTED_BLEND_FILE}" ]]; then
        policy_blend="$(<"${SELECTED_BLEND_FILE}")"
    fi
    echo "Policy reflection blend: ${policy_blend}"
    echo "Headless browser stream port: ${VIEW_PORT}"
    N2_VIEW_CHECKPOINT="${INIT_CHECKPOINT}" \
    N2_STREAM_PORT="${VIEW_PORT}" \
    N2_DEVICE="${TRAIN_DEVICE}" \
    N2_SEED="${TRAIN_SEED}" \
    N2_POLICY_SYMMETRY_BLEND="${policy_blend}" \
        bash humanoid/scripts/run_isaac_stairs_polish.sh view
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
            N2_STABILITY_TRAIN_ITERATIONS=80 \
            N2_STABILITY_CHECKPOINT_INTERVAL=10 \
            N2_STABILITY_EVAL_ENVS=64 \
            N2_STABILITY_HOLDOUT_ENVS=128
        ;;
    long)
        launch_continuous long
        ;;
    diagnose)
        if active_pid >/dev/null; then
            echo "The current training/holdout is still running." >&2
            exit 2
        fi
        require_approved_selection
        run_reflection_preflight
        ;;
    correct|final)
        if active_pid >/dev/null; then
            echo "The current training/holdout is still running; do not start final refinement yet." >&2
            exit 2
        fi
        require_approved_selection
        launch_continuous gait_correction \
            "N2_STABILITY_INIT_CHECKPOINT=${INIT_CHECKPOINT}" \
            N2_STABILITY_SOURCE_APPROVED=True \
            N2_STABILITY_CORRECTION_PREFLIGHT=True \
            N2_STABILITY_TRAIN_ITERATIONS=100 \
            N2_STABILITY_CHECKPOINT_INTERVAL=10 \
            N2_STABILITY_EVAL_ENVS=128 \
            N2_STABILITY_HOLDOUT_ENVS=256 \
            N2_STABILITY_LEARNING_RATE=1.0e-5 \
            N2_STABILITY_ACTION_NOISE=0.04 \
            N2_STABILITY_REFERENCE_COEFF=2.0 \
            N2_STABILITY_SYMMETRIZE_REFERENCE=True \
            N2_STABILITY_SYMMETRY_COEFF=0.0 \
            N2_STABILITY_POLICY_LOSS_SCALE=0.0 \
            N2_STABILITY_ACTOR_LAYERS=4 \
            N2_STABILITY_OBSERVATION_NOISE=0.0 \
            N2_STABILITY_REWARD_OVERRIDES=action_rate=-0.25,action_smoothness=-0.14,dof_acc=-4e-7,stairs_lateral_drift=-18,stairs_heading_alignment=4,stairs_stride_symmetry=-8,stairs_foothold_lateral=2,stairs_foothold_lateral_error=-4,stairs_foot_crossover=-12,stairs_foot_lane_error=-8,stairs_single_support_stability=-4,stairs_right_support_stability=-4,stairs_alternating_tread=2,stairs_repeated_lead=-2,stairs_same_tread_join=-2
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
    view)
        view_selected
        ;;
    *)
        usage
        exit 2
        ;;
esac
