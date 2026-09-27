#!/usr/bin/env bash

set -euo pipefail

# =========================
# 1) 基础路径
# =========================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/default_oven.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# =========================
# 2) 运行参数（可通过同名环境变量覆盖）
# =========================
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESUME_PATH="${RESUME_PATH:-}"   # 续训：ckpt_last.pt
INIT_PATH="${INIT_PATH:-}"       # 阶段 4：从阶段 1 的 ckpt_best.pt 初始化

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

if [[ ! -f "${PROJECT_DIR}/train.py" ]]; then
  echo "错误：未找到训练入口：${PROJECT_DIR}/train.py" >&2
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

if [[ -n "${RESUME_PATH}" && -n "${INIT_PATH}" ]]; then
  echo "错误：RESUME_PATH 与 INIT_PATH 不能同时设置" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

for CKPT in "${RESUME_PATH}" "${INIT_PATH}"; do
  if [[ -n "${CKPT}" && ! -f "${CKPT}" ]]; then
    echo "错误：checkpoint 不存在：${CKPT}" >&2
    exit 1
  fi
done

# =========================
# 4) 日志与 PID
# =========================
DATE_STR="$(date +%y%m%d)"
TIME_STR="$(date +%H%M%S)"
LOG_DIR="${PROJECT_DIR}/logs/oven/${DATE_STR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${TIME_STR}.log"
PID_FILE="${LOG_DIR}/train_${TIME_STR}.pid"

# =========================
# 5) 设备与命令
# =========================
if [[ "${DEVICE}" != "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

CMD=(
  "${PYTHON_BIN}" train.py
  --config "${CONFIG_PATH}"
  --device "${DEVICE}"
)

if [[ -n "${RESUME_PATH}" ]]; then
  CMD+=(--resume "${RESUME_PATH}")
fi
if [[ -n "${INIT_PATH}" ]]; then
  CMD+=(--init "${INIT_PATH}")
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
if [[ -n "${INIT_PATH}" ]]; then
  echo "初始化 checkpoint：${INIT_PATH}"
fi

# =========================
# 6) 后台启动
# =========================
nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "训练已后台启动，PID=${PID}"
echo "日志路径：${LOG_FILE}（完整日志同时写在输出目录的 train.log）"
echo "PID 文件：${PID_FILE}"
