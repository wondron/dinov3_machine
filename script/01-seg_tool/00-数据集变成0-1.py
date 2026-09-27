from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="批量将分割标注图离线转换为 0/1 单通道 mask")
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="mask 根目录，会递归处理其下所有图片",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="是否先备份原文件到同级 _backup 目录",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅检查并打印，不实际写回文件",
    )
    parser.add_argument(
        "--skip_if_binary",
        action="store_true",
        help="如果已经是 0/1，则跳过不写回",
    )
    return parser.parse_args()


def collect_mask_paths(mask_dir: Path) -> list[Path]:
    paths = [
        p for p in mask_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]
    return sorted(paths)


def read_mask_as_gray(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as img:
        arr = np.array(img.convert("L"), dtype=np.uint8)
    return arr


def normalize_mask_to_binary01(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(np.uint8)


def backup_file(src_path: Path, mask_root: Path, backup_root: Path) -> None:
    rel_path = src_path.relative_to(mask_root)
    dst_path = backup_root / rel_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def save_mask(arr: np.ndarray, dst_path: Path) -> None:
    img = Image.fromarray(arr, mode="L")
    img.save(dst_path)


def main():
    args = parse_args()

    mask_dir = Path(args.mask_dir).expanduser().resolve()
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir 不存在: {mask_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"mask_dir 不是目录: {mask_dir}")

    mask_paths = collect_mask_paths(mask_dir)
    if not mask_paths:
        print(f"未在目录中找到可处理的 mask 文件: {mask_dir}")
        return

    backup_root = mask_dir.parent / f"{mask_dir.name}_backup"

    print(f"开始处理，mask 根目录: {mask_dir}")
    print(f"共发现文件数: {len(mask_paths)}")
    print(f"是否备份: {args.backup}")
    print(f"是否仅检查不写回: {args.dry_run}")
    print(f"已是 0/1 是否跳过: {args.skip_if_binary}")
    if args.backup:
        print(f"备份目录: {backup_root}")

    total_count = 0
    changed_count = 0
    skipped_binary_count = 0
    error_count = 0

    for idx, mask_path in enumerate(mask_paths, start=1):
        total_count += 1
        try:
            old_arr = read_mask_as_gray(mask_path)
            old_unique = np.unique(old_arr).tolist()

            new_arr = normalize_mask_to_binary01(old_arr)
            new_unique = np.unique(new_arr).tolist()

            already_binary = set(old_unique).issubset({0, 1})
            if already_binary and args.skip_if_binary:
                skipped_binary_count += 1
                print(
                    f"[{idx}/{len(mask_paths)}] 跳过(已是0/1): {mask_path} | unique={old_unique}"
                )
                continue

            changed = not np.array_equal(old_arr, new_arr)

            print(
                f"[{idx}/{len(mask_paths)}] 处理: {mask_path} | "
                f"原始unique={old_unique} -> 转换后unique={new_unique}"
            )

            if not args.dry_run:
                if args.backup:
                    backup_file(mask_path, mask_dir, backup_root)

                save_mask(new_arr, mask_path)

            if changed:
                changed_count += 1

        except Exception as e:
            error_count += 1
            print(f"[ERROR] 处理失败: {mask_path} | error={e}")

    print("\n========== 处理完成 ==========")
    print(f"总文件数: {total_count}")
    print(f"发生实际内容变化的文件数: {changed_count}")
    print(f"跳过(已是0/1)文件数: {skipped_binary_count}")
    print(f"失败文件数: {error_count}")
    print(f"输出格式: 单通道 L, 像素值 0/1")


if __name__ == "__main__":
    main()