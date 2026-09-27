#!/usr/bin/env bash

set -euo pipefail

# =========================
# 1) 基础路径
# =========================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/default_seg.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# =========================
# 2) 运行参数（可通过同名环境变量覆盖）
# =========================
GPU_ID="${GPU_ID:-5}"
EXP_NAME="${EXP_NAME:-seg}"
DEBUG_STEPS="${DEBUG_STEPS:-0}"
LORA_WEIGHTS="${LORA_WEIGHTS:-}"

# =========================
# 3) 启动前检查
# =========================
if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "错误：项目目录不存在：${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "错误：配置文件不存在：${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/train_seg.py" ]]; then
  echo "错误：未找到分割训练入口：${PROJECT_DIR}/train_seg.py" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "错误：未找到 Python 命令：${PYTHON_BIN}" >&2
  exit 1
fi

if ! [[ "${DEBUG_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "错误：DEBUG_STEPS 必须是大于等于 0 的整数，当前值：${DEBUG_STEPS}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [[ -n "${LORA_WEIGHTS}" && ! -f "${LORA_WEIGHTS}" ]]; then
  echo "错误：LoRA 权重文件不存在：${LORA_WEIGHTS}" >&2
  exit 1
fi

# =========================
# 4) 日志与 PID
# =========================
DATE_STR="$(date +%y%m%d)"
TIME_STR="$(date +%H%M%S)"
LOG_DIR="${PROJECT_DIR}/logs/01-segment/${DATE_STR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TIME_STR}.log"
PID_FILE="${LOG_DIR}/${EXP_NAME}_${TIME_STR}.pid"

# =========================
# 5) 构建命令
# =========================
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

CMD=(
  "${PYTHON_BIN}" train_seg.py
  --config "${CONFIG_PATH}"
  --exp_name "${EXP_NAME}"
)

if (( DEBUG_STEPS > 0 )); then
  CMD+=(--debug_steps "${DEBUG_STEPS}")
fi

if [[ -n "${LORA_WEIGHTS}" ]]; then
  CMD+=(--lora_weights "${LORA_WEIGHTS}")
fi

echo "启动时间：$(date)"
echo "项目目录：${PROJECT_DIR}"
echo "配置文件：${CONFIG_PATH}"
echo "实验名称：${EXP_NAME}"
echo "日志文件：${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# =========================
# 6) 后台启动
# =========================
nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "分割训练已后台启动，PID=${PID}"
echo "日志路径：${LOG_FILE}"
echo "PID 文件：${PID_FILE}"
