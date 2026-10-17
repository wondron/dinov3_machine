"""外部内腔、确认无与未标注的训练语义。"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dino_finetune.config import load_config, validate_config
from dino_finetune.data import OvenDataset
from dino_finetune.device import DeviceSpec
from dino_finetune.labels import LabelSchema, check_label_rules, parse_annotation


class ExternalLabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "sample.json"
        self.path.with_suffix(".jpg").write_bytes(b"image-placeholder")
        self.schema = LabelSchema(["纸质容器"], ["烤盘", "遮挡"], ["无"])
        self.profile = {"公司A": DeviceSpec("公司A", 3, False, "G3", ("烤盘",))}

    def parse(self, annotations):
        self.path.write_text(json.dumps({
            "annotation_version": "1.6", "annotations": annotations,
        }, ensure_ascii=False), encoding="utf-8")
        return parse_annotation(self.path, self.schema)

    def dataset(self, labels):
        return OvenDataset(
            labels, transform=None, container_classes=self.schema.container_classes,
            accessory_classes=self.schema.accessory_classes, profile=self.profile,
            group_names=["G3"], max_rack=8,
        )

    def test_external_annotations_keep_attributes_without_a_profile(self):
        for token in ("未知", "未知型号"):
            with self.subTest(token=token):
                label, issues = self.parse({
                    "device_model": token, "food_name": "板栗", "container_type": ["纸质容器"],
                    "accessory_type": ["烤盘"], "rack_level": ["无"],
                })
                self.assertIsNotNone(label)
                self.assertEqual(issues, [])
                self.assertTrue(label.is_oven)
                self.assertTrue(label.is_external)
                self.assertFalse(label.device_known)
                self.assertEqual(label.device_model, token)
                self.assertTrue(label.food_exist)
                self.assertEqual(label.container, ["纸质容器"])
                self.assertEqual(label.accessory, ["烤盘"])
                self.assertIsNone(label.rack_level)
                self.assertEqual(check_label_rules(label, {}), [])

    def test_external_rack_is_not_validated_against_company_specs(self):
        label, _ = self.parse({
            "device_model": "未知", "food_name": "板栗", "container_type": [],
            "accessory_type": [], "rack_level": 999,
        })
        self.assertEqual(label.rack_level, 999)
        self.assertEqual(check_label_rules(label, self.profile), [])
        self.assertFalse(self.dataset([label]).rack_known[0])

    def test_external_tokens_do_not_allow_arbitrary_misspelled_models(self):
        label, _ = self.parse({"device_model": "公司A_typo"})
        self.assertIn("unknown_device", [issue.code for issue in check_label_rules(label, self.profile)])
        label, _ = self.parse({"device_model": "未知", "is_oven": False})
        self.assertIn("device_not_oven", [issue.code for issue in check_label_rules(label, self.profile)])

    def test_explicit_nulls_are_negatives_but_missing_fields_are_unlabeled(self):
        for food_key in ("food_name", "food_exist"):
            with self.subTest(food_key=food_key):
                label, issues = self.parse({
                    "device_model": "未知", food_key: None,
                    "container_type": None, "accessory_type": None,
                })
                self.assertEqual(issues, [])
                self.assertIs(label.food_exist, False)
                self.assertEqual((label.container, label.accessory), ([], []))
                data = self.dataset([label])
                self.assertTrue(data.food_known[0])
                self.assertTrue(data.container_known[0])
                self.assertTrue(data.accessory_known[0])
                self.assertEqual(float(data.food[0]), 0.0)
                self.assertEqual(float(data.container.sum() + data.accessory.sum()), 0.0)
        label, _ = self.parse({"device_model": "未知"})
        self.assertIsNone(label.food_exist)
        self.assertIsNone(label.container)
        self.assertIsNone(label.accessory)
        data = self.dataset([label])
        self.assertFalse(data.food_known[0])
        self.assertFalse(data.container_known[0])
        self.assertFalse(data.accessory_known[0])

    def test_occlusion_does_not_become_a_negative_or_positive(self):
        for food in ("遮挡", ["遮挡"], ["板栗", "遮挡"]):
            with self.subTest(food=food):
                label, _ = self.parse({
                    "device_model": "未知", "food_name": food,
                    "container_type": "遮挡", "accessory_type": ["烤盘", "遮挡"],
                })
                self.assertIsNotNone(label)
                self.assertIsNone(label.food_exist)
                self.assertIsNone(label.container)
                self.assertIsNone(label.accessory)

    def test_invalid_multilabel_types_report_errors_instead_of_throwing(self):
        for value in (1, True, 1.5, {"遮挡": True}):
            with self.subTest(value=value):
                label, issues = self.parse({"device_model": "未知", "container_type": value})
                self.assertIsNone(label)
                self.assertIn("bad_type", [issue.code for issue in issues])

    def test_dataset_keeps_scene_target_independent_from_device_group(self):
        external, _ = self.parse({
            "device_model": "未知", "food_name": "板栗", "container_type": ["纸质容器"],
            "accessory_type": ["烤盘"], "rack_level": 999,
        })
        known, _ = self.parse({"device_model": "公司A", "rack_level": 2})
        non_oven, _ = self.parse({"is_oven": False, "device_model": None})
        data = self.dataset([external, known, non_oven])
        np.testing.assert_array_equal(data.is_oven, [1, 1, 0])
        np.testing.assert_array_equal(data.group_id, [-2, 0, -1])
        np.testing.assert_array_equal(data.rack_known, [False, True, False])
        self.assertEqual(int(data.rack_count[1]), 3)
        self.assertFalse(data.floor_usable[1])
        self.assertEqual(data.container[0].tolist(), [1.0])
        self.assertEqual(data.accessory[0].tolist(), [1.0, 0.0])

    def test_config_defaults_external_quota_and_rejects_invalid_values(self):
        config = load_config()
        config["sampler"].pop("unknown_oven_ratio")
        validate_config(config)
        self.assertEqual(config["sampler"]["unknown_oven_ratio"], 0.25)
        for value in (-0.1, 1.0, True, "invalid"):
            with self.subTest(value=value):
                config["sampler"]["unknown_oven_ratio"] = value
                with self.assertRaisesRegex(ValueError, "unknown_oven_ratio"):
                    validate_config(config)

    def test_external_tokens_cannot_be_configured_as_none(self):
        for token in ("未知", "未知型号"):
            with self.subTest(token=token):
                config = load_config()
                config["labels"]["none_tokens"].append(token)
                with self.assertRaisesRegex(ValueError, "外部设备标记"):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()
