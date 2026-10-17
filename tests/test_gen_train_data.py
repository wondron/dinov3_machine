"""训练集生成器的标注语义、文件安全及真实入库回归。"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from dino_finetune.config import load_config
from dino_finetune.data import OvenDataset, OvenTransforms
from dino_finetune.device import load_device_profile
from dino_finetune.labels import LabelSchema, load_split, parse_annotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "script" / "01-genTrainData.py"
SPEC = importlib.util.spec_from_file_location("gen_train_data_under_test", SCRIPT_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class _QuietProgress:
    def __new__(cls, iterable, **kwargs):
        return iterable

    @staticmethod
    def write(message):
        pass


class GenerateTrainDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-gen-data-", dir=PROJECT_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / "source"
        self.dst = self.root / "output"
        self.src.mkdir()
        progress = patch.object(GENERATOR, "tqdm", _QuietProgress)
        progress.start()
        self.addCleanup(progress.stop)
        success, encoded = cv2.imencode(".png", np.zeros((3, 5, 3), dtype=np.uint8))
        self.assertTrue(success)
        self.png = encoded.tobytes()

    def write_sample(self, stem, annotations, *, directory=None, nested=True, image_name=None):
        directory = directory or self.src
        directory.mkdir(parents=True, exist_ok=True)
        image_name = image_name or f"{stem}.png"
        image = directory / image_name
        image.write_bytes(self.png)
        raw = {"annotation_version": "1.6", "annotations": copy.deepcopy(annotations)} if nested else copy.deepcopy(annotations)
        raw.update({"image": image_name, "image_name": image_name, "image_root": "/old/machine/path"})
        json_path = directory / f"{stem}.json"
        json_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return json_path

    def test_user_external_sample_keeps_attributes_and_source_unchanged(self):
        raw = {
            "image": "200_1e92b950a54b66b1398edc51a6e0c8fb.jpg",
            "image_name": "200_1e92b950a54b66b1398edc51a6e0c8fb.jpg",
            "image_root": "/data/oridata/随机/其余",
            "annotation_version": "1.6",
            "updated_at": "2026-10-17T05:49:47.417Z",
            "annotations": {
                "food_name": "板栗", "food_count": 45, "quality": None,
                "water_quality": None, "device_model": "未知",
                "container_type": ["纸质容器"], "accessory_type": ["烤盘"],
                "rack_level": ["无"], "food_size": None,
            },
        }
        before = copy.deepcopy(raw)
        normalized = GENERATOR.normalize_annotation(raw)
        ann = normalized["annotations"]
        self.assertIs(ann["is_oven"], True)
        self.assertEqual(ann["device_model"], "未知")
        self.assertIs(ann["food_exist"], True)
        self.assertEqual(ann["container_type"], ["纸质容器"])
        self.assertEqual(ann["accessory_type"], ["烤盘"])
        self.assertIsNone(ann["rack_level"])
        self.assertEqual(ann["food_count"], 45)
        self.assertEqual(normalized["updated_at"], raw["updated_at"])
        self.assertEqual(raw, before)

    def test_both_external_tokens_are_interior_and_never_supervise_rack(self):
        for device in ("未知", "未知型号"):
            with self.subTest(device=device):
                ann = GENERATOR.normalize_annotation({"device_model": device, "rack_level": 3})
                self.assertEqual(ann["device_model"], "未知")
                self.assertIs(ann["is_oven"], True)
                self.assertIsNone(ann["rack_level"])

    def test_nested_and_flat_annotations_have_matching_training_fields(self):
        original = {"device_model": "C9277A", "food_name": "板栗", "container_type": ["纸质容器"],
                    "accessory_type": ["烤盘"], "rack_level": ["3"]}
        flat = GENERATOR.normalize_annotation(original)
        nested = GENERATOR.normalize_annotation({"annotations": original})["annotations"]
        self.assertEqual(flat, nested)
        self.assertEqual(flat["rack_level"], 3)
        self.assertIs(flat["food_exist"], True)

    def test_explicit_null_is_confirmed_absence_but_missing_fields_stay_missing(self):
        for food_field in ("food_name", "food_exist"):
            with self.subTest(food_field=food_field):
                ann = GENERATOR.normalize_annotation({
                    "device_model": "未知", food_field: None,
                    "container_type": None, "accessory_type": None,
                })
                self.assertIs(ann["food_exist"], False)
                self.assertEqual(ann["container_type"], [])
                self.assertEqual(ann["accessory_type"], [])
        ann = GENERATOR.normalize_annotation({"device_model": "未知"})
        for field in ("food_exist", "food_name", "container_type", "accessory_type"):
            self.assertNotIn(field, ann)

    def test_occlusion_is_preserved_as_unlabeled_through_actual_parser(self):
        source = self.write_sample("occluded", {
            "device_model": "未知", "food_name": ["板栗", "遮挡"],
            "container_type": "遮挡", "accessory_type": ["烤盘", "遮挡"],
        })
        GENERATOR.copy_dataset(GENERATOR.collect_pairs(self.src), self.dst)
        schema = LabelSchema(["纸质容器"], ["烤盘"], ["无"])
        label, issues = parse_annotation(self.dst / source.name, schema)
        self.assertIsNotNone(label)
        self.assertFalse(any(issue.level == "error" for issue in issues))
        self.assertIsNone(label.food_exist)
        self.assertIsNone(label.container)
        self.assertIsNone(label.accessory)

    def test_no_rack_does_not_become_floor_and_explicit_zero_is_preserved(self):
        for value in (None, ["无"], "无", []):
            with self.subTest(value=value):
                ann = GENERATOR.normalize_annotation({"device_model": "C9277A", "rack_level": value})
                self.assertIsNone(ann["rack_level"])
        for value in (0, "0", ["0"], 0.0):
            with self.subTest(value=value):
                ann = GENERATOR.normalize_annotation({"device_model": "C9277A", "rack_level": value})
                self.assertEqual(ann["rack_level"], 0)
                self.assertIs(type(ann["rack_level"]), int)

    def test_invalid_json_missing_images_and_invalid_annotation_are_skipped(self):
        valid = self.write_sample("valid", {"device_model": "未知"})
        (self.src / "broken.json").write_text("{bad json", encoding="utf-8")
        (self.src / "missing.json").write_text('{"device_model": "未知"}', encoding="utf-8")
        (self.src / "not-object.json").write_text("[]", encoding="utf-8")
        (self.src / "bad-annotations.json").write_text('{"annotations": []}', encoding="utf-8")
        pairs = GENERATOR.collect_pairs(self.src)
        self.assertEqual([item["json"] for item in pairs], [valid])

    def test_duplicate_stems_are_not_overwritten_and_image_metadata_is_current(self):
        first = self.write_sample("same", {"device_model": "未知"}, directory=self.src / "first")
        second = self.write_sample("same", {"device_model": "C9277A"}, directory=self.src / "second")
        originals = {path: path.read_bytes() for path in (first, second)}
        pairs = GENERATOR.collect_pairs(self.src)
        self.assertEqual(len({item["output_stem"].casefold() for item in pairs}), 2)
        before_copy = copy.deepcopy([item["raw"] for item in pairs])
        GENERATOR.copy_dataset(pairs, self.dst)
        files = list(self.dst.glob("*.json"))
        self.assertEqual(len(files), 2)
        self.assertEqual(len(list(self.dst.glob("*.png"))), 2)
        for output in files:
            raw = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(raw["image"], raw["image_name"])
            self.assertTrue((self.dst / raw["image"]).is_file())
            self.assertEqual(Path(raw["image_root"]), self.dst.resolve())
            self.assertEqual(Path(raw["source_json"]).read_bytes(), originals[Path(raw["source_json"])] )
            self.assertTrue(Path(raw["source_image"]).is_file())
        self.assertEqual([item["raw"] for item in pairs], before_copy)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        with self.assertRaises(FileExistsError):
            GENERATOR.copy_dataset(pairs, self.dst)

    def test_image_lookup_accepts_exported_name_and_uppercase_extension(self):
        self.write_sample("different-annotation", {"device_model": "未知"}, image_name="original.PNG")
        pairs = GENERATOR.collect_pairs(self.src)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["image"].name, "original.PNG")

    def test_image_lookup_respects_relative_subdirectory_before_basename(self):
        json_path = self.write_sample("annotation", {"device_model": "未知"}, image_name="actual.png")
        nested_image = self.src / "images" / "actual.png"
        nested_image.parent.mkdir()
        nested_image.write_bytes(self.png)
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        for relative_name in ("images/actual.png", "images\\actual.png"):
            with self.subTest(relative_name=relative_name):
                raw["image"] = relative_name
                self.assertEqual(GENERATOR.find_image(json_path, raw), nested_image)

    def test_annotations_referencing_one_image_never_cross_splits(self):
        for index in range(6):
            original = self.write_sample(f"image-{index}", {"device_model": "未知"})
            alias = self.src / f"alias-{index}.json"
            alias.write_bytes(original.read_bytes())
        conflict = json.loads((self.src / "image-0.json").read_text(encoding="utf-8"))
        conflict["annotations"]["device_model"] = "C9277A"
        (self.src / "conflicting-model.json").write_text(json.dumps(conflict), encoding="utf-8")
        items = GENERATOR.collect_pairs(self.src)
        splits = GENERATOR.split_dataset(items, seed=17)
        self.assertEqual(sum(map(len, splits)), 13)
        assigned = {}
        for split_index, split in enumerate(splits):
            for item in split:
                image = item["image"].resolve()
                if image in assigned:
                    self.assertEqual(assigned[image], split_index)
                assigned[image] = split_index
        self.assertEqual(len(assigned), 6)

    def test_stratified_split_is_seeded_without_mutation_and_covers_unknown(self):
        items = [{"raw": {"annotations": {"device_model": model}}, "id": f"{model}-{index}",
                  "image": self.src / f"{model}-{index}.png"}
                 for model in ("未知", "C9277A", "C87-i7Pro", None) for index in range(13)]
        before = copy.deepcopy(items)
        splits = GENERATOR.split_dataset(items, seed=712)
        repeated = GENERATOR.split_dataset(items, seed=712)
        self.assertEqual(splits, repeated)
        self.assertEqual(items, before)
        flattened_ids = [item["id"] for split in splits for item in split]
        self.assertEqual(len(flattened_ids), len(set(flattened_ids)))
        self.assertEqual(set(flattened_ids), {item["id"] for item in items})
        for split in splits:
            self.assertIn("未知", {item["raw"]["annotations"]["device_model"] for item in split})
        minimal = [{"raw": {"device_model": "未知"}, "id": index,
                    "image": self.src / f"minimal-{index}.png"} for index in range(3)]
        self.assertEqual([len(split) for split in GENERATOR.split_dataset(minimal)], [1, 1, 1])

    def test_same_day_rerun_preserves_previous_output(self):
        self.write_sample("external", {"device_model": "未知"})
        args = ["--src", str(self.src), "--dst", str(self.dst), "--date", "20261017"]
        GENERATOR.main(args)
        first_output = self.dst / "20261017"
        sentinel = first_output / "keep-me.txt"
        sentinel.write_text("previous result", encoding="utf-8")
        previous = {path.relative_to(first_output): path.read_bytes() for path in first_output.rglob("*") if path.is_file()}
        GENERATOR.main(args)
        runs = sorted(path for path in self.dst.iterdir() if path.is_dir())
        self.assertEqual(len(runs), 2)
        self.assertEqual({path.relative_to(first_output): path.read_bytes() for path in first_output.rglob("*") if path.is_file()}, previous)
        for run in runs:
            self.assertTrue((run / "split_summary.json").is_file())

    def test_generated_splits_load_with_real_profile_and_dataset_masks(self):
        for model in ("C9277A", "C87-i7Pro", "未知"):
            for index in range(3):
                self.write_sample(f"{model}-{index}", {
                    "device_model": model, "food_name": "板栗", "food_count": 45,
                    "container_type": ["纸质容器"], "accessory_type": ["烤盘"],
                    "rack_level": ["无"] if model == "未知" else ["2"],
                    "quality": None, "water_quality": None, "food_size": None,
                }, nested=index != 1)
        before = {path: path.read_bytes() for path in self.src.glob("*.json")}
        GENERATOR.main(["--src", str(self.src), "--dst", str(self.dst), "--date", "20261017"])
        config = load_config(PROJECT_ROOT / "configs" / "default_oven.yaml")
        schema = LabelSchema.from_config(config)
        profile = load_device_profile(PROJECT_ROOT / "configs" / "device_profile.json",
                                      max_rack=config["model"]["max_rack"], accessory_classes=schema.accessory_classes)
        groups = sorted({spec.cavity_group for spec in profile.values()})
        transform = OvenTransforms([8, 8], [0, 0, 0], [1, 1, 1], is_train=False)
        for split in ("train", "val", "test"):
            with self.subTest(split=split):
                labels, issues = load_split([self.dst / "20261017"], split, schema, profile, on_error="raise")
                self.assertEqual(len(labels), 3)
                self.assertFalse(any(issue.code == "unknown_device" for issue in issues))
                dataset = OvenDataset(labels, transform, container_classes=schema.container_classes,
                                      accessory_classes=schema.accessory_classes, profile=profile,
                                      group_names=groups, max_rack=config["model"]["max_rack"])
                external = next(i for i, label in enumerate(labels) if label.device_model == "未知")
                self.assertEqual(float(dataset.is_oven[external]), 1.0)
                self.assertEqual(int(dataset.group_id[external]), -2)
                self.assertFalse(dataset.rack_known[external])
                self.assertTrue(dataset.food_known[external])
                self.assertEqual(float(dataset.food[external]), 1.0)
                self.assertTrue(dataset.container_known[external])
                self.assertTrue(dataset.accessory_known[external])
                self.assertEqual(float(dataset.container[external, schema.container_classes.index("纸质容器")]), 1.0)
                self.assertEqual(float(dataset.accessory[external, schema.accessory_classes.index("烤盘")]), 1.0)
                sample = dataset[external]
                self.assertEqual(tuple(sample["image"].shape), (3, 8, 8))
                self.assertFalse(bool(sample["rack_known"]))
                for index, label in enumerate(labels):
                    if label.device_model != "未知":
                        self.assertGreaterEqual(int(dataset.group_id[index]), 0)
                        self.assertTrue(dataset.rack_known[index])
                        self.assertEqual(int(dataset.rack_level[index]), 2)
        self.assertEqual({path: path.read_bytes() for path in before}, before)


if __name__ == "__main__":
    unittest.main()
