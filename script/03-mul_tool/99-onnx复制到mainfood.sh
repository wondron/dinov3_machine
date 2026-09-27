#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -Eeuo pipefail

# ============================================================
# 配置区域：修改为实际路径
# ============================================================

# 源文件夹
SOURCE_DIR="/data/wangzhuo/01-code/02-project/dinov3_finetune/output/03-multi/260727/onnx"

# 目标文件夹
TARGET_DIR="/data/wangzhuo/01-code/02-project/mainfooddetect/config/model/dino"

# 不需要复制的文件
EXCLUDE_FILE="compare_pt_onnx_multitask.json"

# 不需要复制的文件夹
# 当前脚本本来就不会复制任何文件夹，这里保留用于说明
EXCLUDE_DIR="compare_plots"


# ============================================================
# 路径检查
# ============================================================

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "错误：源文件夹不存在：$SOURCE_DIR" >&2
    exit 1
fi

# 获取规范化后的绝对路径
SOURCE_REAL="$(realpath "$SOURCE_DIR")"
TARGET_REAL="$(realpath -m "$TARGET_DIR")"

if [[ -z "$TARGET_REAL" || "$TARGET_REAL" == "/" ]]; then
    echo "错误：目标文件夹路径不安全：$TARGET_DIR" >&2
    exit 1
fi

if [[ "$SOURCE_REAL" == "$TARGET_REAL" ]]; then
    echo "错误：源文件夹和目标文件夹不能相同。" >&2
    exit 1
fi


# ============================================================
# 创建并清空目标文件夹
# ============================================================

mkdir -p "$TARGET_DIR"

echo "清空目标文件夹：$TARGET_DIR"

# 包括普通文件、隐藏文件和子文件夹
find "$TARGET_DIR" \
    -mindepth 1 \
    -maxdepth 1 \
    -exec rm -rf -- {} +


# ============================================================
# 复制源文件夹当前层级的普通文件
# ============================================================

echo "开始复制文件："
echo "  源文件夹：$SOURCE_DIR"
echo "  目标文件夹：$TARGET_DIR"
echo "  排除文件：$EXCLUDE_FILE"
echo "  不复制任何文件夹，包括：$EXCLUDE_DIR"
echo "------------------------------------------------------------"

copied_count=0

while IFS= read -r -d '' file_path; do
    file_name="$(basename "$file_path")"

    cp -a -- "$file_path" "$TARGET_DIR/"

    echo "已复制：$file_name"
    ((copied_count += 1))
done < <(
    find "$SOURCE_DIR" \
        -mindepth 1 \
        -maxdepth 1 \
        -type f \
        ! -name "$EXCLUDE_FILE" \
        -print0
)

echo "------------------------------------------------------------"
echo "复制完成，共复制 $copied_count 个文件。"
echo "目标文件夹：$TARGET_DIR"