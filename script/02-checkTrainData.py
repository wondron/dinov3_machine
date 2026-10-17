"""Check configured labels, task masks, split leakage and real training batches.

Uses the training data pipeline without constructing a model. Relative paths are
resolved from the project root, just as in train.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from dino_finetune.config import load_config, resolve_path
from dino_finetune.data import build_train_loader
from dino_finetune.device import load_device_profile
from dino_finetune.engine import exclude_groups, group_names_of, make_dataset
from dino_finetune.labels import LabelSchema, check_label_rules, load_split, parse_annotation


def issue_counts(issues) -> dict[str, dict[str, int]]:
    return {
        level: dict(sorted(Counter(issue.code for issue in issues if issue.level == level).items()))
        for level in ("error", "warning")
    }


def diagnose_issues(roots, split, schema, profile, strict):
    """Recover full diagnostics after load_split has rejected a split."""
    issues = []
    for root in roots:
        for path in sorted((root / split).glob("*.json")):
            label, found = parse_annotation(path, schema)
            if label is not None:
                found += check_label_rules(label, profile)
            if strict:
                for issue in found:
                    issue.level = "error"
            issues.extend(found)
    return issues


def task_counts(data) -> dict[str, int]:
    return {
        "is_oven": len(data),
        "proj": int(np.count_nonzero(data.group_id >= 0)),
        **{task: int(np.count_nonzero(getattr(data, task + "_known")))
           for task in ("food", "container", "accessory", "rack")},
    }


def group_counts(group_ids, group_names) -> dict[str, int]:
    names = {-1: "non_oven", -2: "external_oven"}
    names.update(enumerate(group_names))
    return dict(sorted(Counter(names[int(group)] for group in group_ids).items()))


def find_split_leakage(split_labels) -> dict[str, Any]:
    """Hash only image-size buckets present in more than one split."""
    by_size = defaultdict(list)
    for split, labels in split_labels.items():
        for label in labels:
            path = Path(label.image_path).resolve()
            by_size[path.stat().st_size].append({
                "split": split, "image": str(path), "json": label.json_path,
            })
    checked = 0
    leaks = []
    for size, records in by_size.items():
        if len({item["split"] for item in records}) < 2:
            continue
        hashes = {}
        grouped = defaultdict(list)
        for item in records:
            path = item["image"]
            if path not in hashes:
                digest = hashlib.sha256()
                with open(path, "rb") as image:
                    for chunk in iter(lambda: image.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[path] = digest.hexdigest()
                checked += 1
            grouped[hashes[path]].append(item)
        for digest, duplicates in grouped.items():
            if len({item["split"] for item in duplicates}) > 1:
                leaks.append({"sha256": digest, "bytes": size, "samples": duplicates})
    return {"hashed_images": checked, "cross_split_duplicate_groups": len(leaks), "duplicates": leaks}


def check_data(cfg, report, *, batches=1, num_workers=0):
    schema = LabelSchema.from_config(cfg)
    profile = load_device_profile(
        resolve_path(cfg["device_profile"]), max_rack=cfg["model"]["max_rack"],
        accessory_classes=schema.accessory_classes,
    )
    group_names = group_names_of(profile)
    data_cfg = cfg["data"]
    roots = [resolve_path(root) for root in data_cfg["root"]]
    report.update({
        "roots": [str(root) for root in roots], "on_error": "raise",
        "strict": data_cfg["strict"], "num_workers": num_workers,
        "profile_models": sorted(profile), "profile_groups": group_names,
        "splits": {},
    })
    configured_splits = {"train": data_cfg["train_split"], "val": data_cfg["val_split"]}
    if data_cfg["test_split"]:
        configured_splits["test"] = data_cfg["test_split"]
    datasets, split_labels = {}, {}
    for name, split in configured_splits.items():
        info = {
            "split_dir": split, "json_count": sum(len(list((root / split).glob("*.json"))) for root in roots),
            "loaded_count": 0, "dataset_count": 0, "issue_counts": issue_counts([]),
        }
        report["splits"][name] = info
        try:
            labels, issues = load_split(roots, split, schema, profile, on_error="raise", strict=data_cfg["strict"])
            info["loaded_count"] = len(labels)
            labels = exclude_groups(labels, profile, data_cfg["exclude_groups"])
            info["excluded_count"] = info["loaded_count"] - len(labels)
            if not labels:
                raise ValueError(f"{name}: no samples remain after excluding groups")
            dataset = make_dataset(labels, cfg, profile, is_train=name == "train")
            datasets[name], split_labels[name] = dataset, labels
            info.update({
                "status": "passed", "dataset_count": len(dataset), "issue_counts": issue_counts(issues),
                "issues": [asdict(issue) for issue in issues], "task_valid_count": task_counts(dataset),
                "device_model": dict(sorted(Counter(label.device_model or "non_oven" for label in labels).items())),
                "cavity_group": group_counts(dataset.group_id, group_names),
                "transform": dataset.transform.describe(),
            })
        except Exception as exc:
            issues = diagnose_issues(roots, split, schema, profile, data_cfg["strict"])
            info.update({"status": "failed", "error": str(exc), "issue_counts": issue_counts(issues),
                         "issues": [asdict(issue) for issue in issues]})
            report["errors"].append({"stage": f"split:{name}", "type": type(exc).__name__, "message": str(exc)})

    try:
        report["content_leakage"] = find_split_leakage(split_labels)
        if report["content_leakage"]["cross_split_duplicate_groups"]:
            raise ValueError("Identical image content occurs in multiple splits; see content_leakage.duplicates")
    except Exception as exc:
        report["errors"].append({"stage": "content_leakage", "type": type(exc).__name__, "message": str(exc)})

    if "train" not in datasets:
        return
    try:
        dataset = datasets["train"]
        tp = cfg["trainparams"]
        random.seed(tp["seed"])
        np.random.seed(tp["seed"])
        torch.manual_seed(tp["seed"])
        require_supcon = cfg["loss"]["metric"] == "supcon" and cfg["loss"]["weights"]["proj"] > 0
        loader = build_train_loader(
            dataset, batch_size=tp["batch_size"], sampler_cfg=cfg["sampler"],
            steps_per_epoch=tp["steps_per_epoch"] or math.ceil(len(dataset) / tp["batch_size"]),
            num_workers=num_workers, seed=tp["seed"], pin_memory=False, require_supcon=require_supcon,
        )
        sampler = loader.batch_sampler
        report["train_loader"] = {
            "sampler": sampler.describe() if hasattr(sampler, "describe") else type(sampler).__name__,
            "batches_per_epoch": len(loader), "requested_batches": batches, "batches": [],
        }
        for batch in islice(loader, batches):
            images = batch["image"]
            if images.dtype != torch.float32 or not bool(torch.isfinite(images).all()):
                raise ValueError("Training image tensors must be finite float32 values")
            expected = [3, *cfg["input"]["img_dim"]]
            if list(images.shape[1:]) != expected:
                raise ValueError(f"Unexpected training image shape: {list(images.shape)}, expected [B, {expected}]")
            indices = batch["index"].tolist()
            report["train_loader"]["batches"].append({
                "image_shape": list(images.shape), "image_dtype": str(images.dtype),
                "image_min": float(images.min()), "image_max": float(images.max()),
                "cavity_group": group_counts(batch["group_id"].tolist(), group_names),
                "task_valid_count": {
                    "is_oven": len(indices), "proj": int((batch["group_id"] >= 0).sum()),
                    **{task: int(batch[task + "_known"].sum()) for task in ("food", "container", "accessory", "rack")},
                },
                "samples": [{"index": index, "image": dataset.labels[index].image_path,
                             "json": dataset.labels[index].json_path} for index in indices],
            })
        if not report["train_loader"]["batches"]:
            raise ValueError("Training loader yielded no batches")
    except Exception as exc:
        report["errors"].append({"stage": "train_loader", "type": type(exc).__name__, "message": str(exc)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_oven.yaml")
    parser.add_argument("--output", default="output/preflight/data_check.json", help="JSON report path")
    parser.add_argument("--batches", type=int, default=1, help="Real training batches to read (default: 1)")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.batches < 1 or args.num_workers < 0:
        parser.error("--batches must be positive and --num-workers must be nonnegative")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    output = resolve_path(args.output)
    report = {"config": str(resolve_path(args.config)), "status": "failed", "errors": []}
    try:
        cfg = load_config(resolve_path(args.config))
        check_data(cfg, report, batches=args.batches, num_workers=args.num_workers)
    except Exception as exc:
        report["errors"].append({"stage": "setup", "type": type(exc).__name__, "message": str(exc)})
    report["status"] = "failed" if report["errors"] else "passed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, info in report.get("splits", {}).items():
        print(f"{name}: {info['status']}, loaded={info.get('loaded_count', 0)}, "
              f"dataset={info.get('dataset_count', 0)}, issues={info['issue_counts']}")
    for error in report["errors"]:
        print(f"ERROR [{error['stage']}]: {error['message']}", file=sys.stderr)
    print(f"{report['status'].upper()}: {output}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
