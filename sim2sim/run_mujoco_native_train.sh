#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-status}"
PYTHON_BIN="${N2_PYTHON:-}"
TRAIN_DEVICE="${N2_DEVICE:-cuda:0}"
TRAIN_SEED="${N2_SEED:-42}"
INIT_CHECKPOINT="${N2_INIT_CHECKPOINT:-auto}"
NO_WARM_START="${N2_NO_WARM_START:-0}"
MAX_ITERATIONS_OVERRIDE="${N2_MAX_ITERATIONS:-}"
RESUME_ACTION_NOISE_STD="${N2_RESUME_NOISE_STD:-0.18}"
if [[ ! "${TRAIN_SEED}" =~ ^[0-9]+$ ]]; then
    echo "N2_SEED must be a non-negative integer." >&2
    exit 2
fi
if [[ "${NO_WARM_START}" != "0" && "${NO_WARM_START}" != "1" ]]; then
    echo "N2_NO_WARM_START must be 0 or 1." >&2
    exit 2
fi
if [[ -n "${MAX_ITERATIONS_OVERRIDE}" && ! "${MAX_ITERATIONS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "N2_MAX_ITERATIONS must be a positive integer." >&2
    exit 2
fi

WARM_START_ARGS=("--init_checkpoint=${INIT_CHECKPOINT}")
RUN_VARIANT=""
FREEZE_ACTOR_ITERATIONS=100
LEARNING_RATE=2e-5
ACTION_NOISE_STD=0.15
SYMMETRY_LOSS_COEFF=0.75
if [[ "${NO_WARM_START}" == "1" ]]; then
    WARM_START_ARGS=("--no_warm_start")
    RUN_VARIANT="_scratch"
    FREEZE_ACTOR_ITERATIONS=0
    LEARNING_RATE=2e-4
    ACTION_NOISE_STD=0.45
    SYMMETRY_LOSS_COEFF=0.50
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x /root/miniconda3/envs/n2/bin/python ]]; then
        PYTHON_BIN=/root/miniconda3/envs/n2/bin/python
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi

LOG_ROOT=/root/autodl-tmp/n2_train_logs
if [[ ! -d /root/autodl-tmp ]]; then
    LOG_ROOT="${ROOT_DIR}/logs_mujoco/launcher"
fi
mkdir -p "${LOG_ROOT}"
cd "${ROOT_DIR}"

case "${MODE}" in
    smoke)
        exec "${PYTHON_BIN}" -u sim2sim/train_stairs_mujoco.py \
            --smoke \
            "${WARM_START_ARGS[@]}" \
            --device="${TRAIN_DEVICE}" \
            --run_name="mujoco_curriculum_v6_smoke${RUN_VARIANT}_s${TRAIN_SEED}" \
            --seed="${TRAIN_SEED}"
        ;;
    pilot|long|recover|retune)
        if pgrep -f '[t]rain_stairs_mujoco.py' >/dev/null; then
            echo "A MuJoCo stair trainer is already running:"
            pgrep -af '[t]rain_stairs_mujoco.py'
            exit 2
        fi
        STAMP="$(date +%m%d_%H-%M-%S)"
        if [[ "${MODE}" == "pilot" ]]; then
            TARGET_ITERATIONS=800
            RUN_NAME="mujoco_curriculum_v6_pilot${RUN_VARIANT}_s${TRAIN_SEED}"
        elif [[ "${MODE}" == "long" ]]; then
            TARGET_ITERATIONS=3000
            RUN_NAME="mujoco_curriculum_v6_long${RUN_VARIANT}_s${TRAIN_SEED}"
        else
            if [[ "${NO_WARM_START}" == "1" ]]; then
                echo "recover reuses an Actor; set N2_NO_WARM_START=0." >&2
                exit 2
            fi
            RECOVERY_CHECKPOINT="${INIT_CHECKPOINT}"
            if [[ "${RECOVERY_CHECKPOINT}" == "auto" || \
                  -z "${RECOVERY_CHECKPOINT}" || \
                  ! -f "${RECOVERY_CHECKPOINT}" ]]; then
                echo "recover requires an explicit known-good checkpoint:" >&2
                echo "N2_INIT_CHECKPOINT=/absolute/model_rejected.pt $0 recover" >&2
                exit 2
            fi
            WARM_START_ARGS=("--init_checkpoint=${RECOVERY_CHECKPOINT}")
            TARGET_ITERATIONS=600
            RUN_NAME="mujoco_curriculum_v6_recover_s${TRAIN_SEED}"
            FREEZE_ACTOR_ITERATIONS=0
            LEARNING_RATE=3e-5
            ACTION_NOISE_STD=0.18
            SYMMETRY_LOSS_COEFF=0.50
        fi
        if [[ -n "${MAX_ITERATIONS_OVERRIDE}" ]]; then
            TARGET_ITERATIONS="${MAX_ITERATIONS_OVERRIDE}"
        fi
        TRAIN_LOG="${LOG_ROOT}/${RUN_NAME}_${STAMP}.log"
        PID_FILE="${LOG_ROOT}/mujoco_curriculum_v6_s${TRAIN_SEED}.pid"
        nohup "${PYTHON_BIN}" -u sim2sim/train_stairs_mujoco.py \
            "${WARM_START_ARGS[@]}" \
            --num_envs=32 \
            --num_workers=8 \
            --max_iterations="${TARGET_ITERATIONS}" \
            --freeze_actor_iterations="${FREEZE_ACTOR_ITERATIONS}" \
            --rollout_steps=64 \
            --save_interval=50 \
            --selection_interval=50 \
            --selection_episodes=32 \
            --eval_episodes=32 \
            --tournament_candidates=3 \
            --tournament_episodes=32 \
            --learning_rate="${LEARNING_RATE}" \
            --action_noise_std="${ACTION_NOISE_STD}" \
            --symmetry_loss_coeff="${SYMMETRY_LOSS_COEFF}" \
            --critic_symmetry_loss_coeff=0.05 \
            --device="${TRAIN_DEVICE}" \
            --run_name="${RUN_NAME}" \
            --seed="${TRAIN_SEED}" \
            >"${TRAIN_LOG}" 2>&1 </dev/null &
        TRAIN_PID=$!
        printf '%s\n' "${TRAIN_PID}" >"${PID_FILE}"
        echo "Started MuJoCo native training PID=${TRAIN_PID}"
        echo "Mode: ${MODE}, target=${TARGET_ITERATIONS}"
        echo "Log: ${TRAIN_LOG}"
        echo "Watch: sim2sim/run_mujoco_native_train.sh status"
        ;;
    resume)
        if pgrep -f '[t]rain_stairs_mujoco.py' >/dev/null; then
            echo "A MuJoCo stair trainer is already running:"
            pgrep -af '[t]rain_stairs_mujoco.py'
            exit 2
        fi
        LATEST_MODEL="$(find "${ROOT_DIR}/logs_mujoco/n2_stairs_walk" \
            -mindepth 2 -maxdepth 2 -type f \
            -path "*mujoco_curriculum_v[456]_*_s${TRAIN_SEED}*" \
            ! -path '*smoke*' \
            \( -name 'model_[0-9]*.pt' -o -name 'model_interrupted.pt' \) \
            -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -n 1 | cut -d' ' -f2-)"
        if [[ -z "${LATEST_MODEL}" ]]; then
            echo "No native numeric checkpoint was found." >&2
            exit 2
        fi
        RUN_DIR="$(dirname "${LATEST_MODEL}")"
        STAMP="$(date +%m%d_%H-%M-%S)"
        TRAIN_LOG="${LOG_ROOT}/mujoco_curriculum_v6_resume_s${TRAIN_SEED}_${STAMP}.log"
        PID_FILE="${LOG_ROOT}/mujoco_curriculum_v6_s${TRAIN_SEED}.pid"
        TARGET_ITERATIONS="${MAX_ITERATIONS_OVERRIDE:-6000}"
        nohup "${PYTHON_BIN}" -u sim2sim/train_stairs_mujoco.py \
            --resume="${LATEST_MODEL}" \
            --log_dir="${RUN_DIR}" \
            --num_envs=32 \
            --num_workers=8 \
            --max_iterations="${TARGET_ITERATIONS}" \
            --freeze_actor_iterations="${FREEZE_ACTOR_ITERATIONS}" \
            --rollout_steps=64 \
            --save_interval=50 \
            --selection_interval=50 \
            --selection_episodes=32 \
            --eval_episodes=32 \
            --tournament_candidates=3 \
            --tournament_episodes=32 \
            --learning_rate="${LEARNING_RATE}" \
            --action_noise_std="${ACTION_NOISE_STD}" \
            --resume_action_noise_std="${RESUME_ACTION_NOISE_STD}" \
            --symmetry_loss_coeff="${SYMMETRY_LOSS_COEFF}" \
            --critic_symmetry_loss_coeff=0.05 \
            --device="${TRAIN_DEVICE}" \
            --run_name="mujoco_curriculum_v6_resume_s${TRAIN_SEED}" \
            --seed="${TRAIN_SEED}" \
            >"${TRAIN_LOG}" 2>&1 </dev/null &
        TRAIN_PID=$!
        printf '%s\n' "${TRAIN_PID}" >"${PID_FILE}"
        echo "Resumed from: ${LATEST_MODEL}"
        echo "Target iteration: ${TARGET_ITERATIONS}"
        echo "PID=${TRAIN_PID}"
        echo "Log: ${TRAIN_LOG}"
        ;;
    status)
        pgrep -af '[t]rain_stairs_mujoco.py' || true
        LATEST_LOG="$(find "${LOG_ROOT}" -maxdepth 1 -type f \
            -name "mujoco_curriculum_v[456]_*s${TRAIN_SEED}*.log" \
            -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -n 1 | cut -d' ' -f2-)"
        if [[ -n "${LATEST_LOG}" ]]; then
            echo "Latest log: ${LATEST_LOG}"
            tail -n 50 "${LATEST_LOG}"
        fi
        ;;
    stop)
        mapfile -t TRAIN_PIDS < <(pgrep -f '[t]rain_stairs_mujoco.py' || true)
        if [[ "${#TRAIN_PIDS[@]}" -eq 0 ]]; then
            echo "No MuJoCo stair trainer is running."
            exit 0
        fi
        echo "Requesting a safe checkpoint from PID(s): ${TRAIN_PIDS[*]}"
        kill -TERM "${TRAIN_PIDS[@]}"
        for _ in $(seq 1 30); do
            RUNNING=0
            for TRAIN_PID in "${TRAIN_PIDS[@]}"; do
                if kill -0 "${TRAIN_PID}" 2>/dev/null; then
                    RUNNING=1
                fi
            done
            if [[ "${RUNNING}" -eq 0 ]]; then
                echo "Training stopped; model_interrupted.pt was saved."
                exit 0
            fi
            sleep 1
        done
        echo "Trainer is still exiting; check status again shortly." >&2
        exit 1
        ;;
    view)
        LATEST_BEST="$(find "${ROOT_DIR}/logs_mujoco/n2_stairs_walk" \
            -mindepth 2 -maxdepth 2 -type f \
            -path "*mujoco_curriculum_v[456]_*_s${TRAIN_SEED}*" \
            ! -path '*smoke*' \
            -name 'model_best.pt' -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -n 1 | cut -d' ' -f2-)"
        if [[ -z "${LATEST_BEST}" ]]; then
            echo "No accepted v4/v5/v6 model_best.pt exists yet." >&2
            echo "Inspect model_rejected.pt or keep training with resume." >&2
            exit 2
        fi
        export MUJOCO_GL="${MUJOCO_GL:-egl}"
        exec "${PYTHON_BIN}" -u sim2sim/stream_stairs_mujoco.py \
            --checkpoint_path="${LATEST_BEST}" \
            --stream_port=8080 \
            --command_speed=0.18 \
            --seed="${TRAIN_SEED}"
        ;;
    *)
        echo "Usage: $0 {smoke|pilot|long|recover|resume|status|stop|view}" >&2
        exit 2
        ;;
esac
