#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-status}"
PYTHON_BIN="${N2_PYTHON:-}"
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
            --init_checkpoint=auto \
            --device=cuda:0 \
            --run_name=mujoco_curriculum_v2_smoke_s42 \
            --seed=42
        ;;
    pilot|long)
        if pgrep -f '[t]rain_stairs_mujoco.py' >/dev/null; then
            echo "A MuJoCo stair trainer is already running:"
            pgrep -af '[t]rain_stairs_mujoco.py'
            exit 2
        fi
        STAMP="$(date +%m%d_%H-%M-%S)"
        if [[ "${MODE}" == "pilot" ]]; then
            TARGET_ITERATIONS=250
            RUN_NAME=mujoco_curriculum_v2_pilot_s42
        else
            TARGET_ITERATIONS=1200
            RUN_NAME=mujoco_curriculum_v2_long_s42
        fi
        TRAIN_LOG="${LOG_ROOT}/${RUN_NAME}_${STAMP}.log"
        PID_FILE="${LOG_ROOT}/mujoco_curriculum_v2_s42.pid"
        nohup "${PYTHON_BIN}" -u sim2sim/train_stairs_mujoco.py \
            --init_checkpoint=auto \
            --num_envs=32 \
            --num_workers=8 \
            --max_iterations="${TARGET_ITERATIONS}" \
            --freeze_actor_iterations=0 \
            --rollout_steps=96 \
            --save_interval=50 \
            --selection_interval=100 \
            --selection_episodes=16 \
            --eval_episodes=32 \
            --tournament_candidates=3 \
            --tournament_episodes=32 \
            --learning_rate=5e-5 \
            --action_noise_std=0.20 \
            --symmetry_loss_coeff=0.50 \
            --critic_symmetry_loss_coeff=0.05 \
            --device=cuda:0 \
            --run_name="${RUN_NAME}" \
            --seed=42 \
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
            -path '*mujoco_curriculum_v2_*' \
            ! -path '*smoke*' \
            -name 'model_[0-9]*.pt' -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -n 1 | cut -d' ' -f2-)"
        if [[ -z "${LATEST_MODEL}" ]]; then
            echo "No native numeric checkpoint was found." >&2
            exit 2
        fi
        RUN_DIR="$(dirname "${LATEST_MODEL}")"
        STAMP="$(date +%m%d_%H-%M-%S)"
        TRAIN_LOG="${LOG_ROOT}/mujoco_curriculum_v2_resume_${STAMP}.log"
        PID_FILE="${LOG_ROOT}/mujoco_curriculum_v2_s42.pid"
        nohup "${PYTHON_BIN}" -u sim2sim/train_stairs_mujoco.py \
            --resume="${LATEST_MODEL}" \
            --log_dir="${RUN_DIR}" \
            --num_envs=32 \
            --num_workers=8 \
            --max_iterations=1200 \
            --freeze_actor_iterations=0 \
            --rollout_steps=96 \
            --save_interval=50 \
            --selection_interval=100 \
            --selection_episodes=16 \
            --eval_episodes=32 \
            --tournament_candidates=3 \
            --tournament_episodes=32 \
            --learning_rate=5e-5 \
            --action_noise_std=0.20 \
            --symmetry_loss_coeff=0.50 \
            --critic_symmetry_loss_coeff=0.05 \
            --device=cuda:0 \
            --run_name=mujoco_curriculum_v2_resume_s42 \
            --seed=42 \
            >"${TRAIN_LOG}" 2>&1 </dev/null &
        TRAIN_PID=$!
        printf '%s\n' "${TRAIN_PID}" >"${PID_FILE}"
        echo "Resumed from: ${LATEST_MODEL}"
        echo "PID=${TRAIN_PID}"
        echo "Log: ${TRAIN_LOG}"
        ;;
    status)
        pgrep -af '[t]rain_stairs_mujoco.py' || true
        LATEST_LOG="$(find "${LOG_ROOT}" -maxdepth 1 -type f \
            -name 'mujoco_curriculum_v2_*.log' \
            -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -n 1 | cut -d' ' -f2-)"
        if [[ -n "${LATEST_LOG}" ]]; then
            echo "Latest log: ${LATEST_LOG}"
            tail -n 50 "${LATEST_LOG}"
        fi
        ;;
    view)
        export MUJOCO_GL="${MUJOCO_GL:-egl}"
        exec "${PYTHON_BIN}" -u sim2sim/stream_stairs_mujoco.py \
            --checkpoint_path=auto \
            --stream_port=8080 \
            --command_speed=0.18 \
            --seed=42
        ;;
    *)
        echo "Usage: $0 {smoke|pilot|long|resume|status|view}" >&2
        exit 2
        ;;
esac
