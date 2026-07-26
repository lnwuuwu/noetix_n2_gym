#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-}"
SEED="${N2_SEED:-46}"
DEVICE="${N2_DEVICE:-cuda:0}"
NUM_ENVS="${N2_FASTSTAIR_NUM_ENVS:-1024}"
PREFLIGHT_ENVS="${N2_FASTSTAIR_PREFLIGHT_ENVS:-64}"
SCREEN_ENVS="${N2_FASTSTAIR_SCREEN_ENVS:-64}"
HOLDOUT_ENVS="${N2_FASTSTAIR_HOLDOUT_ENVS:-128}"
COMMAND_SPEED="${N2_FASTSTAIR_COMMAND_SPEED:-0.18}"
BASELINE_CHECKPOINT="${N2_FASTSTAIR_BASELINE_CHECKPOINT:-}"
VIEW_PORT="${N2_STREAM_PORT:-18080}"
ALLOW_PHYSX_OVERSUBSCRIPTION="${N2_FASTSTAIR_ALLOW_PHYSX_OVERSUBSCRIPTION:-0}"
SAFE_TRAIN_ENV_LIMIT=1024

CHECKPOINT_INTERVAL="${N2_FASTSTAIR_CHECKPOINT_INTERVAL:-100}"
STAGE1_ITERATIONS="${N2_FASTSTAIR_STAGE1_ITERATIONS:-400}"
STAGE2_ITERATIONS="${N2_FASTSTAIR_STAGE2_ITERATIONS:-600}"
STAGE3_ITERATIONS="${N2_FASTSTAIR_STAGE3_ITERATIONS:-800}"
STAGE1_MIX="${N2_FASTSTAIR_STAGE1_MIX:-0}"
STAGE2_MIX="${N2_FASTSTAIR_STAGE2_MIX:-0,0,1,1,2,2}"
STAGE3_MIX="${N2_FASTSTAIR_STAGE3_MIX:-1,2,2,3,3,4,4,4}"
STAGE1_SPEED="${N2_FASTSTAIR_STAGE1_SPEED:-${COMMAND_SPEED}}"
STAGE2_SPEED="${N2_FASTSTAIR_STAGE2_SPEED:-${COMMAND_SPEED}}"
STAGE3_SPEED="${N2_FASTSTAIR_STAGE3_SPEED:-${COMMAND_SPEED}}"
STAGE1_LEARNING_RATE="${N2_FASTSTAIR_STAGE1_LR:-1.0e-5}"
STAGE2_LEARNING_RATE="${N2_FASTSTAIR_STAGE2_LR:-7.0e-6}"
STAGE3_LEARNING_RATE="${N2_FASTSTAIR_STAGE3_LR:-5.0e-6}"
STAGE1_NOISE="${N2_FASTSTAIR_STAGE1_NOISE:-0.05}"
STAGE2_NOISE="${N2_FASTSTAIR_STAGE2_NOISE:-0.05}"
STAGE3_NOISE="${N2_FASTSTAIR_STAGE3_NOISE:-0.05}"
STAGE1_REFERENCE="${N2_FASTSTAIR_STAGE1_REFERENCE:-0.020}"
STAGE2_REFERENCE="${N2_FASTSTAIR_STAGE2_REFERENCE:-0.010}"
STAGE3_REFERENCE="${N2_FASTSTAIR_STAGE3_REFERENCE:-0.005}"
STAGE1_REWARD_OVERRIDES="${N2_FASTSTAIR_STAGE1_REWARDS:-stairs_alternating_tread=4,stairs_repeated_lead=-3,stairs_same_tread_join=-4,stairs_same_tread_support=-1.5,stairs_sagittal_foot_phase=0.5,stairs_sagittal_foot_phase_error=-0.25,stairs_stride_symmetry=-2,stairs_right_stride_excess=-2,stairs_right_stride_excess_continuous=-1,stairs_foothold_lateral=0.5,stairs_foothold_lateral_error=-0.5,stairs_foot_crossover=-2,stairs_foot_lane_error=-0.5,stairs_single_support_stability=-0.5,stairs_right_support_stability=-0.75,action_rate=-0.10,action_smoothness=-0.10}"
STAGE2_REWARD_OVERRIDES="${N2_FASTSTAIR_STAGE2_REWARDS:-faststair_foothold=8,faststair_foothold_error=-8,stairs_alternating_tread=6,stairs_repeated_lead=-4,stairs_same_tread_join=-6,stairs_same_tread_support=-2,stairs_sagittal_foot_phase=0.75,stairs_sagittal_foot_phase_error=-0.35,stairs_stride_symmetry=-3,stairs_right_stride_excess=-3,stairs_right_stride_excess_continuous=-1.5,stairs_foothold_lateral=0.75,stairs_foothold_lateral_error=-0.75,stairs_foot_crossover=-3,stairs_foot_lane_error=-1,stairs_single_support_stability=-1,stairs_right_support_stability=-1.5,action_rate=-0.12,action_smoothness=-0.12}"
STAGE3_REWARD_OVERRIDES="${N2_FASTSTAIR_STAGE3_REWARDS:-faststair_foothold=10,faststair_foothold_error=-10,stairs_alternating_tread=8,stairs_repeated_lead=-6,stairs_same_tread_join=-8,stairs_same_tread_support=-4,stairs_sagittal_foot_phase=1,stairs_sagittal_foot_phase_error=-0.5,stairs_stride_symmetry=-4,stairs_right_stride_excess=-4,stairs_right_stride_excess_continuous=-2,stairs_foothold_lateral=1,stairs_foothold_lateral_error=-1,stairs_foot_crossover=-4,stairs_foot_lane_error=-1.5,stairs_single_support_stability=-1.5,stairs_right_support_stability=-2.5,action_rate=-0.15,action_smoothness=-0.15}"

LAUNCHER_DIR="${ROOT_DIR}/logs/faststair_launcher"
TRAIN_ROOT="${ROOT_DIR}/logs/n2_faststair"
WORK_ROOT="${LAUNCHER_DIR}/work_s${SEED}"
RESULT_DIR="${LAUNCHER_DIR}/selected_s${SEED}"
PID_FILE="${LAUNCHER_DIR}/faststair_s${SEED}.pid"
LOGPATH_FILE="${LAUNCHER_DIR}/faststair_s${SEED}.logpath"

mkdir -p "${LAUNCHER_DIR}" "${WORK_ROOT}" "${RESULT_DIR}"

CHILD_PID=""
TRAINED_RUN=""
STAGE_SOURCE_ITERATION=""
STAGE_TARGET_ITERATION=""
SCREEN_BEST=""
SCREEN_WINNER=""
SCREEN_DECISION=""
BOOTSTRAP_SOURCE=""

usage() {
    echo "Usage: $0 smoke|train|status|log|stop|view"
    echo "train: Actor-bootstrapped FastStair training (easy -> mixed -> target)."
    echo "The approved legacy PPO checkpoint initializes the Actor and remains the holdout benchmark."
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
    local checkpoint="$1"
    local name
    name="$(basename "${checkpoint}")"
    if [[ "${name}" =~ ^model_([0-9]+)\.pt$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    python -c \
        'import sys,torch; print(int(torch.load(sys.argv[1], map_location="cpu")["iter"]))' \
        "${checkpoint}"
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

cleanup() {
    rm -f "${PID_FILE}"
}

terminate_driver() {
    echo "Received stop signal; forwarding it to the active process."
    if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
        kill -TERM "${CHILD_PID}" 2>/dev/null || true
        wait "${CHILD_PID}" 2>/dev/null || true
    fi
    exit 143
}

latest_run_directory() {
    local run_name="$1"
    find "${TRAIN_ROOT}" -mindepth 1 -maxdepth 1 -type d \
        -name "*_${run_name}" -printf '%T@ %p\n' \
        | sort -nr | head -n 1 | cut -d' ' -f2-
}

resolve_baseline() {
    if [[ -n "${BASELINE_CHECKPOINT}" ]]; then
        if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
            echo "Baseline checkpoint does not exist: ${BASELINE_CHECKPOINT}" >&2
            return 1
        fi
        readlink -f "${BASELINE_CHECKPOINT}"
        return 0
    fi
    local newest=""
    local candidate
    shopt -s nullglob
    for candidate in \
        "${ROOT_DIR}"/logs/isaac_launcher/stability_selected_s*/model_best.pt; do
        if [[ -z "${newest}" || "${candidate}" -nt "${newest}" ]]; then
            newest="${candidate}"
        fi
    done
    shopt -u nullglob
    if [[ -n "${newest}" ]]; then
        readlink -f "${newest}"
    fi
}

materialize_numbered_checkpoint() {
    local source="$1"
    local destination_dir="$2"
    local iteration
    iteration="$(checkpoint_iteration "${source}")"
    local destination="${destination_dir}/model_${iteration}.pt"
    if [[ "$(readlink -f "${source}")" != "$(readlink -m "${destination}")" ]]; then
        cp -f "${source}" "${destination}"
    fi
    echo "${destination}"
}

evaluate_checkpoint() {
    local task="$1"
    local checkpoint="$2"
    local output="$3"
    local env_count="$4"
    local eval_seed="$5"
    local episodes="$6"
    local command_speed="$7"
    local iteration
    iteration="$(checkpoint_iteration "${checkpoint}")"
    run_child python -u humanoid/scripts/eval_stairs.py \
        "--task=${task}" \
        --resume \
        "--load_run=$(dirname "${checkpoint}")" \
        "--checkpoint=${iteration}" \
        --headless \
        "--sim_device=${DEVICE}" \
        "--rl_device=${DEVICE}" \
        "--num_envs=${env_count}" \
        "--seed=${eval_seed}" \
        --terrain_levels=0,1,2,3,4 \
        "--command_speed=${command_speed}" \
        "--episodes_per_env=${episodes}" \
        "--output=${output}"
}

bootstrap_preflight() {
    local baseline="$1"
    local timestamp="$2"
    local work_dir="$3"
    local run_name="faststair_s${SEED}_bootstrap_${timestamp}"
    local run_dir
    local checkpoint
    local candidate_evaluation="${work_dir}/bootstrap_candidate.csv"
    local baseline_evaluation="${work_dir}/bootstrap_baseline.csv"
    local baseline_numbered
    local decision="${work_dir}/bootstrap_decision.json"
    local approved

    echo "FASTSTAIR_BOOTSTRAP_PREFLIGHT envs=${PREFLIGHT_ENVS} speed=${STAGE1_SPEED} checkpoint=${baseline}"
    run_child python -u humanoid/scripts/train.py \
        --task=n2_faststair \
        "--bootstrap_actor_checkpoint=${baseline}" \
        --bootstrap_only \
        --headless \
        "--sim_device=${DEVICE}" \
        "--rl_device=${DEVICE}" \
        "--num_envs=${PREFLIGHT_ENVS}" \
        "--seed=${SEED}" \
        --experiment_name=n2_faststair \
        "--run_name=${run_name}" \
        --terrain_level_mix=0 \
        "--command_speed=${STAGE1_SPEED}" \
        "--action_noise_std=${STAGE1_NOISE}" \
        --freeze_action_noise
    run_dir="$(latest_run_directory "${run_name}")"
    checkpoint="${run_dir}/model_0.pt"
    if [[ -z "${run_dir}" || ! -f "${checkpoint}" ]]; then
        echo "FastStair bootstrap did not save model_0.pt" >&2
        return 2
    fi

    # Compare the migrated Actor and its source on the same deterministic
    # seeds before a single stochastic rollout can enter PPO.
    evaluate_checkpoint \
        n2_faststair "${checkpoint}" "${candidate_evaluation}" \
        "${PREFLIGHT_ENVS}" "${SEED}" 1 "${STAGE1_SPEED}"
    baseline_numbered="$(materialize_numbered_checkpoint \
        "${baseline}" "${work_dir}")"
    evaluate_checkpoint \
        n2_stairs_walk "${baseline_numbered}" "${baseline_evaluation}" \
        "${PREFLIGHT_ENVS}" "${SEED}" 1 "${STAGE1_SPEED}"
    run_child python -u humanoid/scripts/select_faststair_checkpoint.py \
        --preflight \
        "--baseline=${baseline_evaluation}" \
        "--baseline-checkpoint=${baseline}" \
        --candidate \
        "bootstrap_model_0|${candidate_evaluation}|${checkpoint}" \
        "--output=${decision}"
    approved="$(python -c \
        'import json,sys; print(str(json.load(open(sys.argv[1]))["approved"]).lower())' \
        "${decision}")"
    if [[ "${approved}" != "true" ]]; then
        cp -f "${checkpoint}" "${RESULT_DIR}/model_bootstrap_rejected.pt"
        cp -f "${candidate_evaluation}" \
            "${RESULT_DIR}/bootstrap_candidate.csv"
        cp -f "${baseline_evaluation}" \
            "${RESULT_DIR}/bootstrap_baseline.csv"
        cp -f "${decision}" "${RESULT_DIR}/bootstrap_decision.json"
        {
            echo "new_approved=False"
            echo "failure_stage=bootstrap_preflight"
            echo "selected=NONE"
            echo "screen_best=${RESULT_DIR}/model_bootstrap_rejected.pt"
            echo "baseline=${baseline}"
            echo "work_dir=${work_dir}"
        } > "${RESULT_DIR}/search_summary.txt"
        echo "FASTSTAIR_BOOTSTRAP_GATE passed=False decision=${decision}"
        return 1
    fi
    BOOTSTRAP_SOURCE="${checkpoint}"
    echo "FASTSTAIR_BOOTSTRAP_GATE passed=True checkpoint=${checkpoint}"
}

train_stage() {
    local stage="$1"
    local source="$2"
    local extra_iterations="$3"
    local terrain_mix="$4"
    local run_name="$5"
    local source_iteration=0
    local target_iteration
    local resume_options=()
    local command_speed
    local learning_rate
    local action_noise
    local reference_coefficient
    local reward_overrides

    if [[ -z "${source}" || ! -f "${source}" ]]; then
        echo "FastStair stage ${stage} requires a preflight-approved source" >&2
        return 2
    fi
    source_iteration="$(checkpoint_iteration "${source}")"
    resume_options=(
        --resume
        "--load_run=$(dirname "${source}")"
        "--checkpoint=${source_iteration}"
        --reset_optimizer
    )
    case "${stage}" in
        1)
            command_speed="${STAGE1_SPEED}"
            learning_rate="${STAGE1_LEARNING_RATE}"
            action_noise="${STAGE1_NOISE}"
            reference_coefficient="${STAGE1_REFERENCE}"
            reward_overrides="${STAGE1_REWARD_OVERRIDES}"
            ;;
        2)
            command_speed="${STAGE2_SPEED}"
            learning_rate="${STAGE2_LEARNING_RATE}"
            action_noise="${STAGE2_NOISE}"
            reference_coefficient="${STAGE2_REFERENCE}"
            reward_overrides="${STAGE2_REWARD_OVERRIDES}"
            ;;
        3)
            command_speed="${STAGE3_SPEED}"
            learning_rate="${STAGE3_LEARNING_RATE}"
            action_noise="${STAGE3_NOISE}"
            reference_coefficient="${STAGE3_REFERENCE}"
            reward_overrides="${STAGE3_REWARD_OVERRIDES}"
            ;;
        *)
            echo "Unsupported FastStair stage: ${stage}" >&2
            return 2
            ;;
    esac
    target_iteration=$((source_iteration + extra_iterations))
    echo "FASTSTAIR_STAGE_TRAIN stage=${stage} source=${source_iteration} target=${target_iteration} mix=${terrain_mix} speed=${command_speed} lr=${learning_rate} noise=${action_noise} reference=${reference_coefficient}"
    echo "FASTSTAIR_GAIT_REWARDS stage=${stage} ${reward_overrides}"
    run_child python -u humanoid/scripts/train.py \
        --task=n2_faststair \
        "${resume_options[@]}" \
        --headless \
        "--sim_device=${DEVICE}" \
        "--rl_device=${DEVICE}" \
        "--num_envs=${NUM_ENVS}" \
        "--seed=${SEED}" \
        "--max_iterations=${target_iteration}" \
        --experiment_name=n2_faststair \
        "--run_name=${run_name}" \
        "--terrain_level_mix=${terrain_mix}" \
        "--command_speed=${command_speed}" \
        "--learning_rate=${learning_rate}" \
        --fixed_learning_rate \
        "--action_noise_std=${action_noise}" \
        --freeze_action_noise \
        "--actor_reference_loss_coeff=${reference_coefficient}" \
        --actor_policy_loss_scale=0.50 \
        "--reward_scale_overrides=${reward_overrides}" \
        "--save_interval=${CHECKPOINT_INTERVAL}"
    local run_dir
    run_dir="$(latest_run_directory "${run_name}")"
    if [[ -z "${run_dir}" || ! -f "${run_dir}/model_${target_iteration}.pt" ]]; then
        echo "FastStair stage ${stage} did not save model_${target_iteration}.pt" >&2
        return 1
    fi
    TRAINED_RUN="${run_dir}"
    STAGE_SOURCE_ITERATION="${source_iteration}"
    STAGE_TARGET_ITERATION="${target_iteration}"
}

screen_stage() {
    local stage="$1"
    local run_dir="$2"
    local source_iteration="$3"
    local target_iteration="$4"
    local stage_dir="$5"
    local command_speed="$6"
    local candidate_args=()
    local checkpoint
    local iteration
    local output
    mkdir -p "${stage_dir}"
    while IFS= read -r checkpoint; do
        iteration="$(checkpoint_iteration "${checkpoint}")"
        if ((
            ( stage == 1 && iteration < source_iteration )
            || ( stage != 1 && iteration <= source_iteration )
            || iteration > target_iteration
        )); then
            continue
        fi
        if (( iteration != target_iteration && iteration % CHECKPOINT_INTERVAL != 0 )); then
            continue
        fi
        output="${stage_dir}/checkpoint_${iteration}.csv"
        echo "FASTSTAIR_SCREEN stage=${stage} iteration=${iteration}"
        evaluate_checkpoint \
            n2_faststair "${checkpoint}" "${output}" \
            "${SCREEN_ENVS}" "${SEED}" 1 "${command_speed}"
        candidate_args+=(
            --candidate
            "iter_${iteration}|${output}|${checkpoint}"
        )
    done < <(
        find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V
    )
    if (( ${#candidate_args[@]} == 0 )); then
        echo "No screenable checkpoint found in ${run_dir}" >&2
        return 1
    fi
    local decision="${stage_dir}/screen_decision.json"
    SCREEN_DECISION="${decision}"
    run_child python -u humanoid/scripts/select_faststair_checkpoint.py \
        "--stage=${stage}" \
        "${candidate_args[@]}" \
        "--output=${decision}" || return 2
    SCREEN_BEST="$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["screen_best"]["checkpoint"])' \
        "${decision}")"
    local approved
    approved="$(python -c \
        'import json,sys; print(str(json.load(open(sys.argv[1]))["approved"]).lower())' \
        "${decision}")"
    if [[ "${approved}" != "true" ]]; then
        echo "FASTSTAIR_STAGE_GATE stage=${stage} passed=False diagnostic_best=${SCREEN_BEST}"
        return 1
    fi
    SCREEN_WINNER="$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["checkpoint"])' \
        "${decision}")"
    echo "FASTSTAIR_STAGE_GATE stage=${stage} passed=True checkpoint=${SCREEN_WINNER}"
}

run_training() {
    require_positive_integer N2_FASTSTAIR_NUM_ENVS "${NUM_ENVS}"
    require_positive_integer N2_FASTSTAIR_PREFLIGHT_ENVS "${PREFLIGHT_ENVS}"
    require_positive_integer N2_FASTSTAIR_SCREEN_ENVS "${SCREEN_ENVS}"
    require_positive_integer N2_FASTSTAIR_HOLDOUT_ENVS "${HOLDOUT_ENVS}"
    require_positive_integer N2_FASTSTAIR_CHECKPOINT_INTERVAL "${CHECKPOINT_INTERVAL}"
    require_positive_integer N2_FASTSTAIR_STAGE1_ITERATIONS "${STAGE1_ITERATIONS}"
    require_positive_integer N2_FASTSTAIR_STAGE2_ITERATIONS "${STAGE2_ITERATIONS}"
    require_positive_integer N2_FASTSTAIR_STAGE3_ITERATIONS "${STAGE3_ITERATIONS}"
    if (( NUM_ENVS > SAFE_TRAIN_ENV_LIMIT )) \
        && [[ "${ALLOW_PHYSX_OVERSUBSCRIPTION}" != "1" ]]; then
        echo "N2_FASTSTAIR_NUM_ENVS=${NUM_ENVS} exceeds the verified PhysX-safe limit ${SAFE_TRAIN_ENV_LIMIT}." >&2
        echo "The previous 4096-env run exhausted found/lost aggregate-pair buffers and missed contacts." >&2
        echo "Use at most ${SAFE_TRAIN_ENV_LIMIT}; override only for a separately capacity-validated simulator with N2_FASTSTAIR_ALLOW_PHYSX_OVERSUBSCRIPTION=1." >&2
        return 2
    fi
    trap terminate_driver TERM INT
    trap cleanup EXIT

    local timestamp
    timestamp="$(date +%m%d_%H-%M-%S)"
    local work_dir="${WORK_ROOT}/${timestamp}"
    mkdir -p "${work_dir}"
    local source
    local stage
    local iterations
    local mix
    local run_name
    local run_dir
    local source_iteration
    local target_iteration
    local stage_best
    local stage_speed
    local baseline

    baseline="$(resolve_baseline || true)"
    if [[ -z "${baseline}" || ! -f "${baseline}" ]]; then
        echo "FastStair requires an approved n2_stairs_walk checkpoint." >&2
        echo "Set N2_FASTSTAIR_BASELINE_CHECKPOINT=/absolute/path/model_N.pt" >&2
        return 2
    fi

    echo "FASTSTAIR_START seed=${SEED} envs=${NUM_ENVS} planner=dcm_gpu task=n2_faststair"
    echo "FASTSTAIR_PHYSX_GUARD envs=${NUM_ENVS}/${SAFE_TRAIN_ENV_LIMIT} max_gpu_contact_pairs=16777216 buffer_multiplier=8"
    echo "FASTSTAIR_ARCHITECTURE actor_obs=575 critic_obs=217 actor_bootstrap=True critic_bootstrap=False schedule=fixed"
    echo "FASTSTAIR_ANTI_CHEAT planner_forces_opposite_foot=True natural_gait_stage_gates=True"
    echo "FASTSTAIR_BOOTSTRAP checkpoint=${baseline}"
    if ! bootstrap_preflight "${baseline}" "${timestamp}" "${work_dir}"; then
        echo "FASTSTAIR_TRAINING_ABORT reason=bootstrap_preflight_failed"
        echo "No PPO rollout was started; the approved legacy policy is unchanged."
        return 0
    fi
    source="${BOOTSTRAP_SOURCE}"
    for stage in 1 2 3; do
        case "${stage}" in
            1)
                iterations="${STAGE1_ITERATIONS}"
                mix="${STAGE1_MIX}"
                stage_speed="${STAGE1_SPEED}"
                ;;
            2)
                iterations="${STAGE2_ITERATIONS}"
                mix="${STAGE2_MIX}"
                stage_speed="${STAGE2_SPEED}"
                ;;
            3)
                iterations="${STAGE3_ITERATIONS}"
                mix="${STAGE3_MIX}"
                stage_speed="${STAGE3_SPEED}"
                ;;
        esac
        run_name="faststair_s${SEED}_stage${stage}_${timestamp}"
        train_stage \
            "${stage}" "${source}" "${iterations}" "${mix}" \
            "${run_name}"
        run_dir="${TRAINED_RUN}"
        source_iteration="${STAGE_SOURCE_ITERATION}"
        target_iteration="${STAGE_TARGET_ITERATION}"
        if ! screen_stage \
            "${stage}" "${run_dir}" "${source_iteration}" \
            "${target_iteration}" "${work_dir}/stage_${stage}" \
            "${stage_speed}"; then
            if [[ -z "${SCREEN_BEST}" || ! -f "${SCREEN_BEST}" ]]; then
                echo "FastStair screening failed before producing a diagnostic checkpoint." >&2
                return 2
            fi
            cp -f "${SCREEN_BEST}" "${RESULT_DIR}/model_screen_best.pt"
            cp -f "${SCREEN_DECISION}" \
                "${RESULT_DIR}/stage_${stage}_screen_decision.json"
            {
                echo "new_approved=False"
                echo "failure_stage=${stage}"
                echo "selected=NONE"
                echo "screen_best=${RESULT_DIR}/model_screen_best.pt"
                echo "baseline=${baseline}"
                echo "work_dir=${work_dir}"
            } > "${RESULT_DIR}/search_summary.txt"
            echo "FASTSTAIR_STAGE_STOP stage=${stage} reason=promotion_gate_failed"
            echo "N2_FASTSTAIR_SCREEN_BEST=${RESULT_DIR}/model_screen_best.pt"
            echo "No later stage or holdout was run; the approved PPO remains unchanged."
            return 0
        fi
        stage_best="${SCREEN_WINNER}"
        if [[ ! -f "${stage_best}" ]]; then
            echo "Stage selector returned a missing checkpoint: ${stage_best}" >&2
            exit 1
        fi
        source="${stage_best}"
        cp -f "${source}" "${work_dir}/model_stage_${stage}_best.pt"
        echo "FASTSTAIR_STAGE_BEST stage=${stage} checkpoint=${source}"
    done

    local holdout_seed=$((SEED + 1000))
    local candidate_holdout="${work_dir}/holdout_candidate.csv"
    evaluate_checkpoint \
        n2_faststair "${source}" "${candidate_holdout}" \
        "${HOLDOUT_ENVS}" "${holdout_seed}" 2 "${COMMAND_SPEED}"

    local selector_options=()
    local baseline_holdout=""
    local baseline_numbered=""
    if [[ -n "${baseline}" && -f "${baseline}" ]]; then
        baseline_numbered="$(materialize_numbered_checkpoint \
            "${baseline}" "${work_dir}")"
        baseline_holdout="${work_dir}/holdout_baseline.csv"
        echo "FASTSTAIR_BASELINE checkpoint=${baseline}"
        evaluate_checkpoint \
            n2_stairs_walk "${baseline_numbered}" "${baseline_holdout}" \
            "${HOLDOUT_ENVS}" "${holdout_seed}" 2 "${COMMAND_SPEED}"
        selector_options=(
            "--baseline=${baseline_holdout}"
            "--baseline-checkpoint=${baseline}"
        )
    else
        echo "FASTSTAIR_BASELINE checkpoint=NONE absolute_holdout_only=True"
    fi

    local final_decision="${work_dir}/holdout_decision.json"
    run_child python -u humanoid/scripts/select_faststair_checkpoint.py \
        "${selector_options[@]}" \
        --candidate "selected|${candidate_holdout}|${source}" \
        "--output=${final_decision}"
    local approved
    approved="$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["approved"])' \
        "${final_decision}")"
    local selected_iteration
    selected_iteration="$(checkpoint_iteration "${source}")"
    cp -f "${source}" "${RESULT_DIR}/model_screen_best.pt"
    cp -f "${candidate_holdout}" "${RESULT_DIR}/holdout_candidate.csv"
    cp -f "${final_decision}" "${RESULT_DIR}/holdout_decision.json"
    if [[ -n "${baseline_holdout}" ]]; then
        cp -f "${baseline_holdout}" "${RESULT_DIR}/holdout_baseline.csv"
    fi
    if [[ "${approved}" == "True" ]]; then
        cp -f "${source}" "${RESULT_DIR}/model_${selected_iteration}.pt"
        cp -f "${source}" "${RESULT_DIR}/model_best.pt"
        printf '%s\n' "${RESULT_DIR}/model_${selected_iteration}.pt" \
            > "${RESULT_DIR}/selected_checkpoint.txt"
        echo "FASTSTAIR_HOLDOUT_APPROVED=True"
        echo "N2_FASTSTAIR_CHECKPOINT=${RESULT_DIR}/model_${selected_iteration}.pt"
        echo "N2_FASTSTAIR_BEST=${RESULT_DIR}/model_best.pt"
    else
        echo "FASTSTAIR_HOLDOUT_APPROVED=False"
        echo "N2_FASTSTAIR_SCREEN_BEST=${RESULT_DIR}/model_screen_best.pt"
        if [[ -f "${RESULT_DIR}/selected_checkpoint.txt" ]]; then
            echo "N2_FASTSTAIR_CHECKPOINT=$(<"${RESULT_DIR}/selected_checkpoint.txt")"
            echo "The previously approved FastStair model remains untouched."
        else
            echo "The legacy PPO checkpoint remains the approved deployment model."
        fi
    fi
    local preserved_best="NONE"
    if [[ -f "${RESULT_DIR}/selected_checkpoint.txt" ]]; then
        preserved_best="$(<"${RESULT_DIR}/selected_checkpoint.txt")"
    fi
    {
        echo "new_approved=${approved}"
        echo "candidate_iteration=${selected_iteration}"
        echo "selected=${preserved_best}"
        echo "screen_best=${RESULT_DIR}/model_screen_best.pt"
        echo "best=$([[ "${preserved_best}" != "NONE" ]] && echo "${RESULT_DIR}/model_best.pt" || echo NONE)"
        echo "baseline=${baseline:-NONE}"
        echo "evaluation=${RESULT_DIR}/holdout_candidate.csv"
        echo "work_dir=${work_dir}"
    } > "${RESULT_DIR}/search_summary.txt"
}

smoke() {
    local baseline
    local timestamp
    local work_dir
    baseline="$(resolve_baseline || true)"
    if [[ -z "${baseline}" || ! -f "${baseline}" ]]; then
        echo "Smoke test requires N2_FASTSTAIR_BASELINE_CHECKPOINT." >&2
        exit 2
    fi
    require_positive_integer N2_FASTSTAIR_PREFLIGHT_ENVS "${PREFLIGHT_ENVS}"
    timestamp="$(date +%m%d_%H-%M-%S)"
    work_dir="${WORK_ROOT}/smoke_${timestamp}"
    mkdir -p "${work_dir}"
    if bootstrap_preflight "${baseline}" "${timestamp}" "${work_dir}"; then
        echo "FASTSTAIR_SMOKE passed=True checkpoint=${BOOTSTRAP_SOURCE}"
    else
        echo "FASTSTAIR_SMOKE passed=False"
        return 1
    fi
}

view_selected() {
    local checkpoint_file="${RESULT_DIR}/selected_checkpoint.txt"
    if [[ ! -f "${checkpoint_file}" ]]; then
        echo "No independently approved FastStair policy exists for seed ${SEED}." >&2
        echo "Inspect ${RESULT_DIR}/search_summary.txt after training." >&2
        exit 2
    fi
    local checkpoint
    checkpoint="$(<"${checkpoint_file}")"
    local iteration
    iteration="$(checkpoint_iteration "${checkpoint}")"
    python -u humanoid/scripts/stream_stairs.py \
        --task=n2_faststair \
        --resume \
        "--load_run=$(dirname "${checkpoint}")" \
        "--checkpoint=${iteration}" \
        --headless \
        "--sim_device=${DEVICE}" \
        "--rl_device=${DEVICE}" \
        --num_envs=1 \
        "--seed=${SEED}" \
        --terrain_level=4 \
        "--command_speed=${COMMAND_SPEED}" \
        "--stream_port=${VIEW_PORT}"
}

launch() {
    if active_pid >/dev/null; then
        echo "FastStair training is already running PID=$(active_pid)." >&2
        exit 2
    fi
    local timestamp
    timestamp="$(date +%m%d_%H-%M-%S)"
    local log_path="${LAUNCHER_DIR}/faststair_s${SEED}_${timestamp}.log"
    printf '%s\n' "${log_path}" > "${LOGPATH_FILE}"
    nohup bash "$0" _run > "${log_path}" 2>&1 &
    local pid=$!
    printf '%s\n' "${pid}" > "${PID_FILE}"
    echo "Started FastStair-N2 training PID=${pid}"
    echo "Log: ${log_path}"
    echo "Watch: bash humanoid/scripts/run_faststair_n2.sh log"
}

case "${MODE}" in
    smoke)
        smoke
        ;;
    train)
        launch
        ;;
    _run)
        run_training
        ;;
    status)
        if pid="$(active_pid)"; then
            echo "FastStair-N2 training is running PID=${pid}"
        else
            echo "No FastStair-N2 trainer is running."
        fi
        if [[ -f "${RESULT_DIR}/search_summary.txt" ]]; then
            cat "${RESULT_DIR}/search_summary.txt"
        fi
        ;;
    log)
        if [[ ! -f "${LOGPATH_FILE}" ]]; then
            echo "No FastStair log has been recorded." >&2
            exit 2
        fi
        tail -f "$(<"${LOGPATH_FILE}")"
        ;;
    stop)
        if pid="$(active_pid)"; then
            kill -TERM "${pid}"
            echo "Sent SIGTERM to FastStair trainer PID=${pid}."
        else
            echo "No FastStair-N2 trainer is running."
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
