"""Exercise the standalone data check with real images and training labels."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import torch
import yaml

from dino_finetune.config import PROJECT_ROOT, load_config

SPEC = importlib.util.spec_from_file_location("check_train_data", PROJECT_ROOT / "script/02-checkTrainData.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CheckTrainDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = load_config()
        self.cfg["data"].update(root=[str(self.root)], on_error="skip", exclude_groups=[])
        self.cfg["input"]["img_dim"] = [16, 16]
        self.cfg["trainparams"].update(batch_size=8, steps_per_epoch=1)
        self.cfg["sampler"]["p_groups"] = 2
        self.config_path = self.root / "config.yaml"
        self.report_path = self.root / "report.json"
        models = ["C9277A", "C9277A", "C87-i7Pro", "C87-i7Pro", "\u672a\u77e5", None]
        for index, model in enumerate(models):
            self.write_sample("train", index, model, index + 1)
        self.write_sample("val", 0, "C9277A", 30)
        self.write_sample("test", 0, "C87-i7Pro", 40)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def write_sample(self, split, index, model, value):
        directory = self.root / split
        directory.mkdir(exist_ok=True)
        path = directory / f"sample-{index}.json"
        ok, image = cv2.imencode(".png", np.full((20, 24, 3), value, dtype=np.uint8))
        self.assertTrue(ok)
        path.with_suffix(".png").write_bytes(image.tobytes())
        known = model in ("C9277A", "C87-i7Pro")
        path.write_text(json.dumps({"annotations": {
            "is_oven": model is not None, "device_model": model, "food_exist": True,
            "container_type": [], "accessory_type": [self.cfg["labels"]["accessory"][5]],
            "rack_level": 1 if known else None,
        }}), encoding="utf-8")

    def run_check(self):
        self.config_path.write_text(yaml.safe_dump(self.cfg, allow_unicode=True), encoding="utf-8")
        code = CHECKER.main(["--config", str(self.config_path), "--output", str(self.report_path)])
        return code, json.loads(self.report_path.read_text(encoding="utf-8"))

    def test_real_batch_and_task_masks_without_a_model(self):
        code, report = self.run_check()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["splits"]["train"]["task_valid_count"], {
            "is_oven": 6, "proj": 4, "food": 6, "container": 6, "accessory": 6, "rack": 4,
        })
        self.assertEqual(report["on_error"], "raise")
        self.assertEqual(report["num_workers"], 0)
        batches = report["train_loader"]["batches"]
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["image_shape"], [8, 3, 16, 16])
        self.assertEqual(sorted(batches[0]["cavity_group"].values()), [2, 2, 2, 2])

    def test_content_duplicate_across_splits_fails_with_paths(self):
        shutil.copyfile(self.root / "train/sample-0.png", self.root / "val/sample-0.png")
        code, report = self.run_check()
        self.assertEqual(code, 1)
        leakage = report["content_leakage"]
        self.assertEqual(leakage["cross_split_duplicate_groups"], 1)
        self.assertEqual({sample["split"] for sample in leakage["duplicates"][0]["samples"]}, {"train", "val"})
        self.assertTrue(any(error["stage"] == "content_leakage" for error in report["errors"]))

    def test_all_invalid_split_still_writes_issue_details_and_fails(self):
        path = self.root / "val/sample-0.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["annotations"]["container_type"] = ["unlisted-container"]
        path.write_text(json.dumps(raw), encoding="utf-8")
        code, report = self.run_check()
        self.assertEqual(code, 1)
        val = report["splits"]["val"]
        self.assertEqual(val["status"], "failed")
        self.assertEqual(val["dataset_count"], 0)
        self.assertEqual(val["issue_counts"]["error"], {"unknown_class": 1})
        self.assertEqual(val["issues"][0]["json_path"], str(path))


if __name__ == "__main__":
    unittest.main()
