#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-}"
SEED="${N2_SEED:-46}"
DEVICE="${N2_DEVICE:-cuda:0}"
NUM_ENVS="${N2_FASTSTAIR_NUM_ENVS:-4096}"
SCREEN_ENVS="${N2_FASTSTAIR_SCREEN_ENVS:-64}"
HOLDOUT_ENVS="${N2_FASTSTAIR_HOLDOUT_ENVS:-128}"
CHECKPOINT_INTERVAL="${N2_FASTSTAIR_CHECKPOINT_INTERVAL:-200}"
COMMAND_SPEED="${N2_FASTSTAIR_COMMAND_SPEED:-0.18}"
BASELINE_CHECKPOINT="${N2_FASTSTAIR_BASELINE_CHECKPOINT:-}"
VIEW_PORT="${N2_STREAM_PORT:-18080}"

STAGE1_ITERATIONS="${N2_FASTSTAIR_STAGE1_ITERATIONS:-1200}"
STAGE2_ITERATIONS="${N2_FASTSTAIR_STAGE2_ITERATIONS:-1600}"
STAGE3_ITERATIONS="${N2_FASTSTAIR_STAGE3_ITERATIONS:-1200}"
STAGE1_MIX="${N2_FASTSTAIR_STAGE1_MIX:-0,0,0,1,1,2,2,3}"
STAGE2_MIX="${N2_FASTSTAIR_STAGE2_MIX:-0,1,1,2,2,3,3,4}"
STAGE3_MIX="${N2_FASTSTAIR_STAGE3_MIX:-1,2,2,3,3,4,4,4}"

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

usage() {
    echo "Usage: $0 smoke|train|status|log|stop|view"
    echo "train: from-scratch FastStair safety pretraining (easy -> mixed -> target)."
    echo "The legacy PPO checkpoint is used only as the final safety benchmark."
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
        "--command_speed=${COMMAND_SPEED}" \
        "--episodes_per_env=${episodes}" \
        "--output=${output}"
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

    if [[ -n "${source}" ]]; then
        source_iteration="$(checkpoint_iteration "${source}")"
        resume_options=(
            --resume
            "--load_run=$(dirname "${source}")"
            "--checkpoint=${source_iteration}"
            --reset_optimizer
        )
    fi
    target_iteration=$((source_iteration + extra_iterations))
    echo "FASTSTAIR_STAGE_TRAIN stage=${stage} source=${source_iteration} target=${target_iteration} mix=${terrain_mix}"
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
        --learning_rate=3.0e-4 \
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
    local candidate_args=()
    local checkpoint
    local iteration
    local output
    mkdir -p "${stage_dir}"
    while IFS= read -r checkpoint; do
        iteration="$(checkpoint_iteration "${checkpoint}")"
        if (( iteration <= source_iteration || iteration > target_iteration )); then
            continue
        fi
        if (( iteration != target_iteration && iteration % CHECKPOINT_INTERVAL != 0 )); then
            continue
        fi
        output="${stage_dir}/checkpoint_${iteration}.csv"
        echo "FASTSTAIR_SCREEN stage=${stage} iteration=${iteration}"
        evaluate_checkpoint \
            n2_faststair "${checkpoint}" "${output}" \
            "${SCREEN_ENVS}" "${SEED}" 1
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
    run_child python -u humanoid/scripts/select_faststair_checkpoint.py \
        "${candidate_args[@]}" \
        "--output=${decision}"
    SCREEN_BEST="$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["screen_best"]["checkpoint"])' \
        "${decision}")"
    echo "FASTSTAIR_SCREEN_BEST stage=${stage} checkpoint=${SCREEN_BEST}"
}

run_training() {
    require_positive_integer N2_FASTSTAIR_NUM_ENVS "${NUM_ENVS}"
    require_positive_integer N2_FASTSTAIR_SCREEN_ENVS "${SCREEN_ENVS}"
    require_positive_integer N2_FASTSTAIR_HOLDOUT_ENVS "${HOLDOUT_ENVS}"
    require_positive_integer N2_FASTSTAIR_CHECKPOINT_INTERVAL "${CHECKPOINT_INTERVAL}"
    require_positive_integer N2_FASTSTAIR_STAGE1_ITERATIONS "${STAGE1_ITERATIONS}"
    require_positive_integer N2_FASTSTAIR_STAGE2_ITERATIONS "${STAGE2_ITERATIONS}"
    require_positive_integer N2_FASTSTAIR_STAGE3_ITERATIONS "${STAGE3_ITERATIONS}"
    trap terminate_driver TERM INT
    trap cleanup EXIT

    local timestamp
    timestamp="$(date +%m%d_%H-%M-%S)"
    local work_dir="${WORK_ROOT}/${timestamp}"
    mkdir -p "${work_dir}"
    local source=""
    local stage
    local iterations
    local mix
    local run_name
    local run_dir
    local source_iteration
    local target_iteration
    local stage_best

    echo "FASTSTAIR_START seed=${SEED} envs=${NUM_ENVS} planner=dcm_gpu task=n2_faststair"
    echo "FASTSTAIR_ARCHITECTURE actor_obs=575 critic_obs=217 warm_start=False schedule=adaptive"
    for stage in 1 2 3; do
        case "${stage}" in
            1)
                iterations="${STAGE1_ITERATIONS}"
                mix="${STAGE1_MIX}"
                ;;
            2)
                iterations="${STAGE2_ITERATIONS}"
                mix="${STAGE2_MIX}"
                ;;
            3)
                iterations="${STAGE3_ITERATIONS}"
                mix="${STAGE3_MIX}"
                ;;
        esac
        run_name="faststair_s${SEED}_stage${stage}_${timestamp}"
        train_stage "${stage}" "${source}" "${iterations}" "${mix}" "${run_name}"
        run_dir="${TRAINED_RUN}"
        source_iteration="${STAGE_SOURCE_ITERATION}"
        target_iteration="${STAGE_TARGET_ITERATION}"
        screen_stage \
            "${stage}" "${run_dir}" "${source_iteration}" \
            "${target_iteration}" "${work_dir}/stage_${stage}"
        stage_best="${SCREEN_BEST}"
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
        "${HOLDOUT_ENVS}" "${holdout_seed}" 2

    local selector_options=()
    local baseline
    baseline="$(resolve_baseline || true)"
    local baseline_holdout=""
    local baseline_numbered=""
    if [[ -n "${baseline}" && -f "${baseline}" ]]; then
        baseline_numbered="$(materialize_numbered_checkpoint \
            "${baseline}" "${work_dir}")"
        baseline_holdout="${work_dir}/holdout_baseline.csv"
        echo "FASTSTAIR_BASELINE checkpoint=${baseline}"
        evaluate_checkpoint \
            n2_stairs_walk "${baseline_numbered}" "${baseline_holdout}" \
            "${HOLDOUT_ENVS}" "${holdout_seed}" 2
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
    python -u humanoid/scripts/train.py \
        --task=n2_faststair \
        --headless \
        "--sim_device=${DEVICE}" \
        "--rl_device=${DEVICE}" \
        --num_envs=64 \
        "--seed=${SEED}" \
        --max_iterations=5 \
        --experiment_name=n2_faststair_smoke \
        --run_name=planner_smoke \
        --terrain_level_mix=0,1,2,3,4 \
        --save_interval=5
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
