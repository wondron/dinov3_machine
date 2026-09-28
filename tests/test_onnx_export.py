"""固定单张 ONNX 导出、数值对齐和目录推理的回归测试。"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from dino_finetune.device import DeviceGallery, DeviceSpec
from dino_finetune.metrics import default_calibration
from dino_finetune.model.oven import OUTPUT_KEYS, OvenMultiTaskModel, inference_outputs

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "script" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_script = load_script("export_onnx_test", "6-export_onnx.py")
infer_script = load_script("infer_onnx_test", "7-infer_onnx.py")


class TinyEncoder(nn.Module):
    num_features = 16

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, self.num_features, kernel_size=4, stride=4)
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.qkv = nn.Linear(self.num_features, 3 * self.num_features)
        self.blocks = nn.ModuleList([block])

    def forward_features(self, images):
        tokens = self.patch(images).flatten(2).transpose(1, 2)
        q, k, v = self.blocks[0].attn.qkv(tokens).chunk(3, dim=-1)
        tokens = tokens + torch.tanh(q + k + v)
        return {"x_norm_clstoken": tokens.mean(1), "x_norm_patchtokens": tokens}


@unittest.skipUnless(
    importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
    "需要安装 onnx 和 onnxruntime",
)
class SingleImageExportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = OvenMultiTaskModel(
            TinyEncoder(), num_container=2, num_accessory=2, max_rack=3,
            attn_heads=2, hidden_dim=8, proj_hidden_dim=8, proj_dim=6,
            dropout=0, use_lora=True, lora_rank=2,
        ).eval()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "linear_b_" in name:
                    param.normal_(std=0.02)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_fixed_shape_and_single_image_numerical_equivalence(self):
        import onnxruntime as ort

        images = torch.randn(3, 3, 16, 16)
        expected = export_script.single_image_outputs(self.model, images)
        self.model.merge_lora()
        model_path = self.directory / "oven.onnx"
        export_script.export_onnx(self.model, images[:1], model_path, 18)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.assertEqual(session.get_inputs()[0].shape, [1, 3, 16, 16])
        self.assertTrue(all(output.shape[0] == 1 for output in session.get_outputs()))
        for i, image in enumerate(images):
            actual = dict(zip(OUTPUT_KEYS, session.run(None, {"image": image[None].numpy()})))
            comparison = export_script.compare({key: value[i:i + 1] for key, value in expected.items()}, actual)
            self.assertTrue(all(entry["ok"] for entry in comparison.values()), comparison)

    def test_rejects_export_with_batch_larger_than_one(self):
        model_path = self.directory / "invalid.onnx"
        with self.assertRaisesRegex(ValueError, r"\[1,3,H,W\]"):
            export_script.export_onnx(self.model, torch.randn(2, 3, 16, 16), model_path, 18)
        self.assertFalse(model_path.exists())

    def make_run(self):
        inp = {"img_dim": [16, 16], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "img_interp": "linear"}
        schema = SimpleNamespace(container_classes=["bowl", "foil"], accessory_classes=["tray", "rack"])
        profile = {"A": DeviceSpec("A", 3, True, "A", ("tray", "rack"))}
        cal = default_calibration(
            {"default_threshold": 0.5, "gallery_feature": "proj", "knn_k": 1, "tau": 0.5},
            schema.container_classes, schema.accessory_classes,
        )
        cal["fingerprint"] = "test-version"
        galleries = {key: DeviceGallery({"A": "A"}) for key in ("proj", "cls")}
        with torch.no_grad():
            output = inference_outputs(self.model(torch.zeros(1, 3, 16, 16)))
        for key, gallery in galleries.items():
            gallery.add("A", output[key])
        (self.directory / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")
        (self.directory / "device_profile.json").write_text(json.dumps({k: v.to_dict() for k, v in profile.items()}), encoding="utf-8")
        images = self.directory / "images"
        images.mkdir()
        rng = np.random.default_rng(42)
        for i in range(3):
            cv2.imwrite(str(images / f"{i}.png"), rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
        run = SimpleNamespace(
            model=self.model, run_dir=self.directory, cfg={"input": inp, "model": {"max_rack": 3}},
            schema=schema, profile=profile, calibration=cal, galleries=galleries,
            gallery_meta={"fingerprint": "test-version"},
        )
        return run, images

    def test_export_checks_multiple_images_then_infers_directory_one_at_a_time(self):
        run, images = self.make_run()
        argv = ["export", "--run", str(self.directory), "--check_input", str(images), "--device", "cpu"]
        batch_sizes = []
        handle = self.model.register_forward_pre_hook(lambda _, args: batch_sizes.append(args[0].shape[0]))
        try:
            with patch.object(export_script, "load_run", return_value=run), patch("sys.argv", argv):
                export_script.main()
        finally:
            handle.remove()
        self.assertTrue(batch_sizes)
        self.assertTrue(all(size == 1 for size in batch_sizes))
        bundle = self.directory / "onnx"
        check = json.loads((bundle / "export_check.json").read_text(encoding="utf-8"))
        self.assertTrue(check["passed"])
        self.assertEqual((check["batch"], check["num_images"]), (1, 3))
        meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["input"]["batch_size"], 1)
        result_path = self.directory / "predictions.json"
        argv = ["infer", "--onnx_dir", str(bundle), "--input", str(images), "--out", str(result_path), "--provider", "cpu"]
        with patch("sys.argv", argv):
            infer_script.main()
        predictions = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(len(predictions), 3)
        self.assertEqual([Path(p["image"]).name for p in predictions], ["0.png", "1.png", "2.png"])

    def test_failed_merge_check_preserves_existing_bundle(self):
        run, images = self.make_run()
        bundle = self.directory / "onnx"
        bundle.mkdir()
        previous = {"oven.onnx": b"previous valid model", "meta.json": b"previous metadata", "export_check.json": b"previous check"}
        for name, contents in previous.items():
            (bundle / name).write_bytes(contents)
        bad_comparison = {key: {"ok": False} for key in OUTPUT_KEYS}
        argv = ["export", "--run", str(self.directory), "--check_input", str(images), "--device", "cpu"]
        good_comparison = {key: {"ok": True} for key in OUTPUT_KEYS}
        with patch.object(export_script, "load_run", return_value=run), patch("sys.argv", argv), \
                patch.object(export_script, "compare", side_effect=[bad_comparison, good_comparison]):
            with self.assertRaisesRegex(RuntimeError, "导出校验未通过"):
                export_script.main()
        for name, contents in previous.items():
            self.assertEqual((bundle / name).read_bytes(), contents)
        self.assertFalse(json.loads((bundle / "export_check.failed.json").read_text(encoding="utf-8"))["passed"])
        self.assertFalse(list(bundle.glob(".export-*")))


if __name__ == "__main__":
    unittest.main()
