"""真实外部设备用于拒识标定，不能进入公司图库及层位指标。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dino_finetune.device import DeviceGallery, DeviceSpec
from dino_finetune.engine import build_galleries, exclude_groups, select_gallery_indices
from dino_finetune.labels import OvenLabel
from dino_finetune.metrics import calibrate_thresholds, default_calibration, evaluate_outputs


def label(model, index=0):
    return OvenLabel(f"image-{index}.jpg", f"image-{index}.json", model is not None, model, None, None, None, None)


def output_arrays(labels, feats):
    n = len(labels)
    return {
        "index": np.arange(n),
        "is_oven": np.array([lb.is_oven for lb in labels], dtype=np.float32),
        "is_oven_prob": np.full(n, 0.9, dtype=np.float32),
        "food_known": np.zeros(n, dtype=bool), "food": np.zeros(n), "food_prob": np.zeros(n),
        "container_known": np.zeros(n, dtype=bool),
        "container": np.zeros((n, 1)), "container_prob": np.zeros((n, 1)),
        "accessory_known": np.zeros(n, dtype=bool),
        "accessory": np.zeros((n, 1)), "accessory_prob": np.zeros((n, 1)),
        # 即使上游误传外部层位有效，指标与标定仍必须排除它。
        "rack_known": np.ones(n, dtype=bool), "rack_level": np.ones(n, dtype=np.int64),
        "rack_logits": np.tile([0., 5.], (n, 1)), "rack_raw": np.tile([0., 5.], (n, 1)),
        "proj": np.asarray(feats, dtype=np.float32), "cls": np.asarray(feats, dtype=np.float32),
    }


class ExternalMetricsTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "公司A": DeviceSpec("公司A", 1, True, "G1", ()),
            "公司B": DeviceSpec("公司B", 1, True, "G2", ()),
        }
        self.eval_cfg = {
            "default_threshold": 0.5, "gallery_feature": "proj", "knn_k": 1,
            "tau": 0.55, "safety_recall_target": 0.95, "rack_target_precision": 0.95,
        }

    def gallery(self, second_group=False):
        gallery = DeviceGallery({name: spec.cavity_group for name, spec in self.profile.items()})
        gallery.add("公司A", torch.tensor([[1., 0.]]))
        if second_group:
            gallery.add("公司B", torch.tensor([[0., 1.]]))
        return gallery

    def loo_module(self):
        path = Path(__file__).resolve().parents[1] / "script" / "4-留一型号验证.py"
        spec = importlib.util.spec_from_file_location("loo_external_metrics_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def calibrate(self, labels, arrays, gallery):
        return calibrate_thresholds(
            arrays, labels, eval_cfg=self.eval_cfg, container_classes=["容器"],
            accessory_classes=["附件"], safety_classes=[], galleries={"proj": gallery},
        )

    def test_real_external_queries_calibrate_with_one_company_group(self):
        labels = [label("公司A"), label("未知", 1)]
        arrays = output_arrays(labels, [[0.8, 0.6], [0., 1.]])
        # 外部的低层位置信度不能参与层位置信度阈值标定。
        arrays["rack_logits"][1] = [0., 0.]
        calibration = self.calibrate(labels, arrays, self.gallery())
        self.assertEqual(calibration["sources"]["tau/proj"], "external")
        self.assertAlmostEqual(calibration["tau"]["proj"], 0.4, places=6)
        self.assertEqual(calibration["tau_calibration"]["proj"]["n_in"], 1)
        self.assertEqual(calibration["tau_calibration"]["proj"]["n_out"], 1)
        self.assertGreater(calibration["rack_conf_threshold"], 0.99)

    def test_no_external_queries_keep_gallery_loo_fallback(self):
        labels = [label("公司A"), label("公司B", 1)]
        arrays = output_arrays(labels, [[0.8, 0.6], [0.6, 0.8]])
        calibration = self.calibrate(labels, arrays, self.gallery(second_group=True))
        self.assertEqual(calibration["sources"]["tau/proj"], "gallery_loo")
        self.assertAlmostEqual(calibration["tau"]["proj"], 0.7, places=6)
        self.assertEqual(calibration["tau_calibration"]["proj"]["n_out"], 2)

    def test_single_company_group_without_external_keeps_default(self):
        labels = [label("公司A")]
        calibration = self.calibrate(labels, output_arrays(labels, [[1., 0.]]), self.gallery())
        self.assertEqual(calibration["tau"]["proj"], self.eval_cfg["tau"])
        self.assertNotIn("proj", calibration["tau_calibration"])

    def test_external_queries_need_in_gallery_validation_queries(self):
        labels = [label("未知")]
        calibration = self.calibrate(labels, output_arrays(labels, [[0., 1.]]), self.gallery())
        self.assertEqual(calibration["tau"]["proj"], self.eval_cfg["tau"])
        self.assertNotIn("proj", calibration["tau_calibration"])

    def test_external_is_not_selected_or_built_into_gallery(self):
        labels = [label("未知"), label("公司A", 1), label(None, 2), label("未知型号", 3)]
        self.assertEqual(select_gallery_indices(labels, 10, 42), [1])
        self.assertEqual(exclude_groups(labels, self.profile, ["G1"]), [labels[0], labels[2], labels[3]])
        arrays = {
            "index": np.array([3, 1, 0, 2]),
            "proj": np.array([[0., 1.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float32),
        }
        arrays["cls"] = arrays["proj"].copy()
        galleries = build_galleries(arrays, labels, self.profile)
        for gallery in galleries.values():
            self.assertEqual(gallery.labels, ["公司A"])
            torch.testing.assert_close(gallery.bank, torch.tensor([[1., 0.]]))

    def test_external_rejection_metrics_do_not_affect_company_top1_or_rack(self):
        labels = [label("公司A"), label("未知", 1), label("未知型号", 2)]
        arrays = output_arrays(labels, [[1., 0.], [0.9, 0.43589], [0., 1.]])
        calibration = default_calibration(self.eval_cfg, ["容器"], ["附件"])
        flat, report = evaluate_outputs(
            arrays, labels, container_classes=["容器"], accessory_classes=["附件"],
            safety_classes=[], profile=self.profile, galleries={"proj": self.gallery()}, calibration=calibration,
        )
        self.assertEqual(report["device_proj"]["n"], 3)
        self.assertEqual(report["device_proj"]["n_external"], 2)
        self.assertEqual(flat["device_top1"], 1.)
        self.assertEqual(flat["external_reject_rate"], 0.5)
        self.assertEqual(flat["external_false_accept_rate"], 0.5)
        self.assertEqual(report["rack"]["n"], 1)
        self.assertEqual(report["cascade"]["n"], 1)

    def test_loo_split_does_not_treat_external_as_company_group(self):
        module = self.loo_module()
        train = [label("公司A"), label("未知", 1)]
        val = [label("公司A", 2), label("未知型号", 3)]
        refs, out_queries, in_queries = module.split_queries(train, val, self.profile, 0.2, 10, 42)
        self.assertEqual(refs, {"G1": [train[0]]})
        self.assertEqual(out_queries, {"G1": [val[0]]})
        self.assertEqual(in_queries, {"G1": [val[0]]})

    def test_loo_does_not_overwrite_threshold_from_real_external_images(self):
        labels = [label("公司A"), label("未知", 1)]
        calibration = self.calibrate(labels, output_arrays(labels, [[0.8, 0.6], [0., 1.]]), self.gallery())
        report = {
            "features": {"proj": {"tau": 0.99, "tau_calibration": {"tau": 0.99}}},
            "recommended_feature": "proj", "folds": {"G1": {}},
        }
        run_dir = Path("mock-run")
        with patch.object(Path, "read_text", return_value=json.dumps(calibration)), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "write_text") as write:
            self.loo_module().apply_to_calibration(run_dir, report)
        self.assertEqual(write.call_count, 2)  # 原标定备份，然后写回更新结果。
        applied = json.loads(write.call_args.args[0])
        self.assertEqual(applied["tau"]["proj"], calibration["tau"]["proj"])
        self.assertEqual(applied["sources"]["tau/proj"], "external")


if __name__ == "__main__":
    unittest.main()
