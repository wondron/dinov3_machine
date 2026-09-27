#!/usr/bin/env bash

set -euo pipefail

# =========================
# 1) 基础路径
# =========================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/default_cls.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# =========================
# 2) 运行参数（可通过同名环境变量覆盖）
# =========================
GPU_ID="${GPU_ID:-5}"
DEVICE="${DEVICE:-cuda}"
MAX_STEPS="${MAX_STEPS:-0}"

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

if [[ ! -f "${PROJECT_DIR}/train_cls.py" ]]; then
  echo "错误：未找到分类训练入口：${PROJECT_DIR}/train_cls.py" >&2
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

if ! [[ "${MAX_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "错误：MAX_STEPS 必须是大于等于 0 的整数，当前值：${MAX_STEPS}" >&2
  exit 1
fi

# =========================
# 4) 日志与 PID 目录
# =========================
DATE_STR="$(date +%y%m%d)"
TIME_STR="$(date +%H%M%S)"
LOG_DIR="${PROJECT_DIR}/logs/02-class/${DATE_STR}"
LOG_FILE="${LOG_DIR}/cls_${TIME_STR}.log"
PID_FILE="${LOG_DIR}/cls_${TIME_STR}.pid"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

if [[ "${DEVICE}" != "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

CMD=(
  "${PYTHON_BIN}" train_cls.py
  --config "${CONFIG_PATH}"
  --device "${DEVICE}"
)

if (( MAX_STEPS > 0 )); then
  CMD+=(--max_steps "${MAX_STEPS}")
fi

echo "启动时间：$(date)"
echo "项目目录：${PROJECT_DIR}"
echo "配置文件：${CONFIG_PATH}"
echo "日志文件：${LOG_FILE}"
echo "运行设备：${DEVICE}"
if [[ "${DEVICE}" != "cpu" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# =========================
# 5) 后台启动分类训练
# =========================
nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "分类训练已后台启动，PID=${PID}"
echo "日志路径：${LOG_FILE}"
echo "PID 文件：${PID_FILE}"
