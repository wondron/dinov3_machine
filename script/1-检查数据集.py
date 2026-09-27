#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查语义分割数据集：image 与 mask 的 (H, W) 是否一致
- 默认：递归扫描 root 下的常见图片/掩码后缀，并按 "basename" 自动配对
- 输出：终端摘要 + 生成 report.txt / report.csv（可选）
"""

import argparse
import csv
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff"}  # mask 常见以 png/tif 为主


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def is_mask_file(p: Path) -> bool:
    return p.suffix.lower() in MASK_EXTS


def get_hw(p: Path) -> tuple[int, int]:
    # PIL size 是 (W, H)
    with Image.open(p) as im:
        w, h = im.size
    return h, w


def build_index(files: list[Path]) -> dict[str, list[Path]]:
    """
    用 basename（去掉后缀）做 key。一个 key 可能对应多个文件（例如 jpg/png 同名）。
    """
    idx: dict[str, list[Path]] = {}
    for p in files:
        key = p.stem
        idx.setdefault(key, []).append(p)
    return idx


def choose_best(candidates: list[Path], prefer_exts: list[str]) -> Path:
    """
    同 basename 多候选时，按 prefer_exts 优先级选择，否则取最短路径+字典序稳定选择。
    """
    cand_sorted = sorted(candidates, key=lambda x: (len(str(x)), str(x)))
    for ext in prefer_exts:
        for p in cand_sorted:
            if p.suffix.lower() == ext:
                return p
    return cand_sorted[0]


def main():
    ap = argparse.ArgumentParser(description="检查 image/mask 尺寸一致性（语义分割）")
    ap.add_argument("--root", required=True, help="数据集根目录")
    ap.add_argument("--images-dir", default="", help="仅在该子目录下找 image（可选，如 images）")
    ap.add_argument("--masks-dir", default="", help="仅在该子目录下找 mask（可选，如 masks/annotations）")
    ap.add_argument("--report-dir", default="", help="报告输出目录（默认 root）")
    ap.add_argument("--csv", action="store_true", help="额外输出 report.csv")
    ap.add_argument("--limit", type=int, default=0, help="只打印前 N 个问题（0 表示不限制）")
    ap.add_argument("--strict-pair", action="store_true",
                    help="严格配对：只检查同时存在 image+mask 的 key；否则也会报告缺失")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"❌ 错误：root 不存在：{root}")

    img_root = (root / args.images_dir).resolve() if args.images_dir else root
    msk_root = (root / args.masks_dir).resolve() if args.masks_dir else root

    if not img_root.exists():
        raise SystemExit(f"❌ 错误：images-dir 目录不存在：{img_root}")
    if not msk_root.exists():
        raise SystemExit(f"❌ 错误：masks-dir 目录不存在：{msk_root}")

    # 扫描文件
    img_files = [p for p in img_root.rglob("*") if p.is_file() and is_image_file(p)]
    msk_files = [p for p in msk_root.rglob("*") if p.is_file() and is_mask_file(p)]

    # 构建索引（basename -> 文件列表）
    img_idx = build_index(img_files)
    msk_idx = build_index(msk_files)

    # 需要检查的 key 集合
    if args.strict_pair:
        keys = sorted(set(img_idx.keys()) & set(msk_idx.keys()))
    else:
        keys = sorted(set(img_idx.keys()) | set(msk_idx.keys()))

    report_dir = Path(args.report_dir).resolve() if args.report_dir else root
    report_dir.mkdir(parents=True, exist_ok=True)
    report_txt = report_dir / "report.txt"
    report_csv = report_dir / "report.csv"

    # 统计
    total_keys = len(keys)
    missing_img = 0
    missing_msk = 0
    shape_mismatch = 0
    io_errors = 0

    rows_for_csv = []
    lines = []
    lines.append(f"root={root}")
    lines.append(f"images_scan_root={img_root}")
    lines.append(f"masks_scan_root={msk_root}")
    lines.append(f"total_keys={total_keys}")
    lines.append("")

    # 偏好：image 优先 jpg/jpeg；mask 优先 png
    img_prefer = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]
    msk_prefer = [".png", ".tif", ".tiff", ".bmp"]

    printed = 0
    for k in tqdm(keys, desc="Checking image/mask pairs", unit="pair"):
        img_cands = img_idx.get(k, [])
        msk_cands = msk_idx.get(k, [])

        if not img_cands:
            missing_img += 1
            msg = f"[缺失IMAGE] key={k} mask={msk_cands[0] if msk_cands else 'None'}"
            lines.append(msg)
            rows_for_csv.append(["missing_image", k, "", str(msk_cands[0]) if msk_cands else "", "", ""])
            continue

        if not msk_cands:
            missing_msk += 1
            msg = f"[缺失MASK ] key={k} image={img_cands[0]}"
            lines.append(msg)
            rows_for_csv.append(["missing_mask", k, str(img_cands[0]), "", "", ""])
            continue

        img_p = choose_best(img_cands, img_prefer)
        msk_p = choose_best(msk_cands, msk_prefer)

        try:
            ih, iw = get_hw(img_p)
            mh, mw = get_hw(msk_p)
        except Exception as e:
            io_errors += 1
            msg = f"[读取失败] key={k} image={img_p} mask={msk_p} err={repr(e)}"
            lines.append(msg)
            rows_for_csv.append(["io_error", k, str(img_p), str(msk_p), "", repr(e)])
            continue

        if (ih, iw) != (mh, mw):
            shape_mismatch += 1
            msg = f"[尺寸不一致] key={k} image=({ih},{iw}) {img_p} | mask=({mh},{mw}) {msk_p}"
            lines.append(msg)
            rows_for_csv.append(["shape_mismatch", k, str(img_p), str(msk_p), f"{ih}x{iw}", f"{mh}x{mw}"])

            if args.limit == 0 or printed < args.limit:
                printed += 1

    # 摘要
    lines.append("")
    lines.append("==== SUMMARY ====")
    lines.append(f"images_found={len(img_files)} masks_found={len(msk_files)}")
    lines.append(f"missing_image={missing_img}")
    lines.append(f"missing_mask={missing_msk}")
    lines.append(f"shape_mismatch={shape_mismatch}")
    lines.append(f"io_errors={io_errors}")
    lines.append(f"report_txt={report_txt}")
    if args.csv:
        lines.append(f"report_csv={report_csv}")

    report_txt.write_text("\n".join(lines), encoding="utf-8")

    if args.csv:
        with report_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["type", "key", "image_path", "mask_path", "image_hw", "mask_hw_or_err"])
            w.writerows(rows_for_csv)

    # 终端输出摘要（尽量简洁）
    print("\n".join(lines[-10:]))  # 打印最后几行摘要
    if shape_mismatch > 0 or missing_img > 0 or missing_msk > 0 or io_errors > 0:
        print("\n⚠️ 发现问题，详见 report.txt（以及可选 report.csv）。")
    else:
        print("\n✅ 未发现 image/mask 尺寸或缺失问题。")


if __name__ == "__main__":
    main()
