#!/usr/bin/env bash

set -euo pipefail

# =========================
# 1) 基础路径
# =========================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/default_mul.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# =========================
# 2) 运行参数（可通过同名环境变量覆盖）
# =========================
GPU_ID="${GPU_ID:-4}"
DEVICE="${DEVICE:-cuda}"
RESUME_PATH="${RESUME_PATH:-}"

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

if [[ ! -f "${PROJECT_DIR}/train_mul.py" ]]; then
  echo "错误：未找到多任务训练入口：${PROJECT_DIR}/train_mul.py" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "错误：未找到 Python 命令：${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${DEVICE}" != "auto" && "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
  echo "错误：DEVICE 仅支持 auto、cpu 或 cuda，当前值：${DEVICE}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [[ -n "${RESUME_PATH}" && ! -f "${RESUME_PATH}" ]]; then
  echo "错误：续训 checkpoint 不存在：${RESUME_PATH}" >&2
  exit 1
fi

# =========================
# 4) 输出目录
# =========================
DATE_STR="$(date +%y%m%d)"
TIME_STR="$(date +%H%M%S)"
LOG_DIR="${PROJECT_DIR}/logs/03-multi/${DATE_STR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/multitask_${TIME_STR}.log"
PID_FILE="${LOG_DIR}/multitask_${TIME_STR}.pid"

# =========================
# 5) 设备与命令
# =========================
if [[ "${DEVICE}" != "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

CMD=(
  "${PYTHON_BIN}" train_mul.py
  --config "${CONFIG_PATH}"
  --device "${DEVICE}"
)

if [[ -n "${RESUME_PATH}" ]]; then
  CMD+=(--resume "${RESUME_PATH}")
fi

echo "启动时间：$(date)"
echo "项目目录：${PROJECT_DIR}"
echo "配置文件：${CONFIG_PATH}"
echo "日志文件：${LOG_FILE}"
echo "运行设备：${DEVICE}"
if [[ "${DEVICE}" != "cpu" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
if [[ -n "${RESUME_PATH}" ]]; then
  echo "续训 checkpoint：${RESUME_PATH}"
fi

# =========================
# 6) 后台启动
# =========================
nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "多任务训练已后台启动，PID=${PID}"
echo "日志路径：${LOG_FILE}"
echo "PID 文件：${PID_FILE}"
