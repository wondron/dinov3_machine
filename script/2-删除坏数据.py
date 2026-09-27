#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 report.csv 删除 image / mask 坏样本
默认 dry-run（不真正删除）
加 --apply 才会真的删
"""

import argparse
import csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="report.csv 路径")
    ap.add_argument("--apply", action="store_true", help="真的执行删除（危险）")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise SystemExit(f"❌ CSV 不存在: {csv_path}")

    to_delete = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == "shape_mismatch":
                img = row["image_path"]
                msk = row["mask_path"]
                to_delete.append((img, msk))

    print(f"发现 shape_mismatch 样本数: {len(to_delete)}")
    print("-" * 80)

    deleted_img = 0
    deleted_msk = 0

    for img, msk in to_delete:
        img_p = Path(img)
        msk_p = Path(msk)

        if img_p.exists():
            print(f"[IMAGE] {'删除' if args.apply else '将删除'}: {img_p}")
            if args.apply:
                img_p.unlink()
                deleted_img += 1
        else:
            print(f"[IMAGE] 已不存在: {img_p}")

        if msk_p.exists():
            print(f"[MASK ] {'删除' if args.apply else '将删除'}: {msk_p}")
            if args.apply:
                msk_p.unlink()
                deleted_msk += 1
        else:
            print(f"[MASK ] 已不存在: {msk_p}")

        print("-" * 80)

    print("====== SUMMARY ======")
    print(f"image 删除数: {deleted_img}")
    print(f"mask  删除数: {deleted_msk}")

    if not args.apply:
        print("\n⚠️ 当前是 dry-run，如确认无误，请加 --apply 执行真正删除")
    else:
        print("\n✅ 删除完成")

if __name__ == "__main__":
    main()
