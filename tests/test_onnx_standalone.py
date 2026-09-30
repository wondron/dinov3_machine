"""Exercise the copied deployment package with a real, tiny ONNX model."""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ("numpy", "cv2", "onnx", "onnxruntime")
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in DEPENDENCIES)
if HAS_RUNTIME:
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort

OUTPUT_KEYS = (
    "is_oven_prob", "food_prob", "container_prob", "accessory_prob", "rack_raw", "proj", "cls",
)


def fixture_outputs():
    return {
        "is_oven_prob": np.array([0.9], dtype=np.float32),
        "food_prob": np.array([0.8], dtype=np.float32),
        "container_prob": np.array([[0.9, 0.1]], dtype=np.float32),
        "accessory_prob": np.array([[0.8, 0.9]], dtype=np.float32),
        "rack_raw": np.array([[10.0, 0.0, 3.0, 12.0]], dtype=np.float32),
        "proj": np.array([[1.0, 0.0]], dtype=np.float32),
        "cls": np.array([[1.0, 0.0]], dtype=np.float32),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_bundle(directory):
    directory.mkdir()
    meta = {
        "fingerprint": "fixture-v1",
        "input": {
            "name": "image", "batch_size": 1, "img_dim": [2, 3],
            "mean": [0.1, 0.2, 0.3], "std": [0.2, 0.4, 0.5],
            "img_interp": "nearest", "color": "RGB", "layout": "NCHW",
        },
        "outputs": list(OUTPUT_KEYS),
        "container_classes": ["bowl", "foil"],
        "accessory_classes": ["tray", "rack"],
        "max_rack": 3,
    }
    calibration = {
        "fingerprint": "fixture-v1", "gallery_feature": "proj", "knn_k": 1,
        "tau": {"proj": 0.5, "cls": 0.5}, "is_oven_threshold": 0.5,
        "food_threshold": 0.5, "container_thresholds": {"bowl": 0.5, "foil": 0.5},
        "accessory_thresholds": {"tray": 0.5, "rack": 0.5}, "rack_conf_threshold": 0.6,
    }
    profile = {
        "A": {"rack_count": 2, "floor_usable": False, "accessories": ["tray"], "cavity_group": "A"},
        "B": {"rack_count": 3, "floor_usable": True, "accessories": None, "cavity_group": "B"},
    }
    gallery = {
        "meta": {"fingerprint": "fixture-v1"}, "group_of": {"A": "A", "B": "B"},
        "features": {"proj": {"file": "gallery_proj.npy", "labels": ["A", "B"]}},
    }
    for filename, value in (
        ("meta.json", meta), ("calibration.json", calibration),
        ("device_profile.json", profile), ("gallery.json", gallery),
    ):
        write_json(directory / filename, value)
    np.save(directory / "gallery_proj.npy", np.eye(2, dtype=np.float32))

    # The graph order deliberately differs from meta.json: outputs must be requested by name.
    outputs = fixture_outputs()
    nodes, graph_outputs = [], []
    for name in reversed(OUTPUT_KEYS):
        value = outputs[name]
        nodes.append(onnx.helper.make_node(
            "Constant", [], [name], value=onnx.numpy_helper.from_array(value),
        ))
        graph_outputs.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape))
    graph = onnx.helper.make_graph(
        nodes, "standalone-test", [onnx.helper.make_tensor_value_info(
            "image", onnx.TensorProto.FLOAT, [1, 3, 2, 3],
        )], graph_outputs,
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(directory / "oven.onnx"))
    return meta, calibration


@unittest.skipUnless(HAS_RUNTIME, "Requires numpy, opencv-python-headless, onnx and onnxruntime")
class StandaloneOnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.external = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.external.cleanup)
        cls.code = Path(cls.external.name) / "deploy"
        shutil.copytree(ROOT / "script" / "onnx_code", cls.code, ignore=shutil.ignore_patterns("__pycache__"))
        cls.package_name = "_standalone_deploy_test"
        spec = importlib.util.spec_from_file_location(
            cls.package_name, cls.code / "__init__.py", submodule_search_locations=[str(cls.code)],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[cls.package_name] = package
        cls.addClassCleanup(cls.remove_test_modules)
        spec.loader.exec_module(package)
        cls.module = importlib.import_module(cls.package_name + ".c_onnx_classify")
        cls.post_module = importlib.import_module(cls.package_name + ".onnx_postprocess")

    @classmethod
    def remove_test_modules(cls):
        for name in list(sys.modules):
            if name == cls.package_name or name.startswith(cls.package_name + "."):
                del sys.modules[name]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name) / "bundle"
        self.meta, self.calibration = make_bundle(self.bundle)
        self.image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3) * 3

    def classifier(self, **overrides):
        config = {"onnx_dir": str(self.bundle), "provider": "cpu", **overrides}
        return self.module.OnnxClassifier({"CLASS_CONFIG": config})

    def postprocessor(self, calibration=None):
        galleries, _ = self.post_module.load_gallery_bundle(self.bundle)
        profile = self.post_module.load_device_profile(
            self.bundle / "device_profile.json", max_rack=3, accessory_classes=["tray", "rack"],
        )
        return self.post_module.OvenPostprocessor(
            calibration=calibration or self.calibration, profile=profile,
            container_classes=["bowl", "foil"], accessory_classes=["tray", "rack"], galleries=galleries,
        )

    def test_external_project_import_and_inference_without_training_dependencies(self):
        program = """
import builtins, json, sys
real_import = builtins.__import__
def deployment_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'dino_finetune', 'app'}:
        raise AssertionError('deployment imports training/application dependency: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = deployment_import
sys.path.insert(0, sys.argv[1])
import numpy as np
from c_onnx_classify import OnnxClassifier
classifier = OnnxClassifier({'CLASS_CONFIG': {'model_path': sys.argv[2], 'provider': 'cpu'}})
print(json.dumps(classifier.detect(np.zeros((4, 6, 3), dtype=np.uint8))))
"""
        process = subprocess.run(
            [sys.executable, "-I", "-c", program, str(self.code), str(self.bundle / "oven.onnx")],
            cwd=self.temp.name, text=True, encoding="utf-8", capture_output=True, timeout=60,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["device_model"], "A")
        self.assertEqual(result["rack_level"], 2)

    def test_preprocess_resize_color_order_normalization_and_input_forms(self):
        classifier = self.classifier()
        expected = self.image[::2, ::2, ::-1].astype(np.float32) / np.float32(255.0)
        expected = (expected - np.array([0.1, 0.2, 0.3], dtype=np.float32)) / np.array(
            [0.2, 0.4, 0.5], dtype=np.float32,
        )
        expected = expected.transpose(2, 0, 1)[None]
        ok, encoded = cv2.imencode(".png", self.image)
        self.assertTrue(ok)
        path = Path(self.temp.name) / "测试图.png"
        path.write_bytes(encoded.tobytes())
        for image in (self.image, encoded.tobytes(), path, str(path)):
            with self.subTest(input_type=type(image).__name__):
                actual = classifier._preprocess(image)
                self.assertEqual(actual.shape, (1, 3, 2, 3))
                self.assertEqual(actual.dtype, np.dtype("float32"))
                self.assertTrue(actual.flags.c_contiguous)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
                self.assertEqual(classifier.detect(image)["device_model"], "A")

    def test_named_outputs_and_single_result_match_expected_decisions(self):
        classifier = self.classifier(with_scores=True)
        self.assertTrue(classifier.isInit)
        self.assertEqual(
            [item.name for item in classifier.session.get_outputs()], list(reversed(OUTPUT_KEYS)),
        )
        actual = classifier.detect(self.image)
        self.assertEqual(actual, {
            "is_oven": True, "device_model": "A", "device_score": 1.0,
            "food_exist": True, "container_type": ["bowl"], "accessory_type": ["tray"],
            "rack_level": 2, "rack_level_score": 0.9526, "low_confidence": False,
            "scores": {"is_oven": 0.9, "food": 0.8, "container": {"bowl": 0.9, "foil": 0.1},
                       "accessory": {"tray": 0.8, "rack": 0.9}},
        })
        self.assertNotIn("scores", classifier.detect(self.image, with_scores=False))

    def test_model_path_and_device_override(self):
        classifier = self.module.OnnxClassifier({"CLASS_CONFIG": {
            "model_path": str(self.bundle / "oven.onnx"), "provider": "cpu", "device_model": "B",
        }})
        result = classifier.detect(self.image)
        self.assertEqual(result["device_model"], "B")
        self.assertEqual(result["accessory_type"], ["tray", "rack"])
        self.assertEqual(result["rack_level"], 3)
        self.assertEqual(classifier.detect(self.image, "A")["rack_level"], 2)
        with self.assertRaises(ValueError):
            classifier.detect(self.image, "missing-model")

    def test_warmup_executes_and_none_does_not_infer(self):
        classifier = self.classifier()
        with patch.object(classifier.session, "run", wraps=classifier.session.run) as run:
            self.assertEqual(classifier.detect(None), {})
            run.assert_not_called()
            classifier.warmup(times=3)
            self.assertEqual(run.call_count, 3)

    def test_malformed_inputs_fail_before_inference(self):
        classifier = self.classifier()
        with patch.object(classifier.session, "run", wraps=classifier.session.run) as run:
            for image in (b"not an image", b"", np.zeros((4, 6), dtype=np.uint8), self.image.astype(np.float32)):
                with self.subTest(input_type=type(image).__name__), self.assertRaises((ValueError, TypeError)):
                    classifier.detect(image)
            run.assert_not_called()

    def test_bundle_fingerprints_must_agree(self):
        for filename in ("meta.json", "calibration.json", "gallery.json"):
            with self.subTest(filename=filename):
                path = self.bundle / filename
                original = json.loads(path.read_text(encoding="utf-8"))
                changed = copy.deepcopy(original)
                target = changed["meta"] if filename == "gallery.json" else changed
                target["fingerprint"] = "different-version"
                write_json(path, changed)
                try:
                    with patch.object(ort, "InferenceSession") as session, self.assertRaises(RuntimeError):
                        self.classifier()
                    session.assert_not_called()
                finally:
                    write_json(path, original)

    def test_explicit_cuda_fails_when_unavailable_and_auto_uses_cpu(self):
        with patch.object(ort, "get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaises(RuntimeError):
                self.classifier(provider="cuda")
            classifier = self.classifier(provider="auto")
            self.assertEqual(classifier.session.get_providers(), ["CPUExecutionProvider"])
            self.assertTrue(classifier.detect(self.image)["is_oven"])

    def test_non_oven_unknown_device_and_empty_cavity(self):
        post = self.postprocessor()
        non_oven = fixture_outputs()
        non_oven["is_oven_prob"][:] = 0.1
        result = post(non_oven, device_model="A")[0]
        self.assertEqual(result["device_model"], "无")
        self.assertEqual(result["accessory_type"], ["tray", "rack"])
        self.assertIsNone(result["rack_level"])
        self.assertNotIn("pending_gallery", result)

        unknown = fixture_outputs()
        unknown["proj"][:] = -1
        result = post(unknown)[0]
        self.assertEqual(result["device_model"], "未知型号")
        self.assertEqual(result["accessory_type"], ["tray", "rack"])
        self.assertTrue(result["pending_gallery"])
        self.assertIsNone(result["rack_level"])
        self.assertAlmostEqual(result["device_score"], -0.7071, places=4)

        empty = fixture_outputs()
        for key in ("food_prob", "container_prob", "accessory_prob"):
            empty[key][:] = 0
        result = post(empty)[0]
        self.assertEqual(result["device_model"], "A")
        self.assertIsNone(result["rack_level"])
        self.assertFalse(result["low_confidence"])

    def test_rack_masks_and_both_low_confidence_conditions(self):
        post = self.postprocessor()
        missing_accessory = fixture_outputs()
        missing_accessory["accessory_prob"][:] = 0
        result = post(missing_accessory, device_model="A")[0]
        self.assertEqual(result["rack_level"], 2)
        self.assertTrue(result["low_confidence"])
        uncertain = fixture_outputs()
        uncertain["rack_raw"][:] = 0
        result = post(uncertain, device_model="A")[0]
        self.assertEqual(result["rack_level"], 1)
        self.assertEqual(result["rack_level_score"], 0.5)
        self.assertTrue(result["low_confidence"])

    def test_gallery_votes_by_cavity_group_and_uses_configured_feature(self):
        index_path = self.bundle / "gallery.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["group_of"] = {"A": "A", "A2": "A", "B": "B"}
        index["features"] = {"cls": {"file": "gallery_cls.npy", "labels": ["B", "A", "A2"]}}
        write_json(index_path, index)
        profile_path = self.bundle / "device_profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["A2"] = dict(profile["A"])
        write_json(profile_path, profile)
        vectors = np.array([[0.9, np.sqrt(0.19)], [0.8, 0.6], [0.7, np.sqrt(0.51)]], dtype=np.float32)
        np.save(self.bundle / "gallery_cls.npy", vectors)
        calibration = {**self.calibration, "gallery_feature": "cls", "knn_k": 3}
        outputs = fixture_outputs()
        outputs["proj"][:] = -1  # Only cls may participate in this retrieval.
        result = self.postprocessor(calibration)(outputs)[0]
        self.assertEqual(result["device_model"], "A")
        self.assertEqual(result["device_score"], 0.625)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional reference parity requires torch")
    def test_matches_training_preprocess_and_postprocessor(self):
        from dino_finetune.data import OvenTransforms
        from dino_finetune.device import load_device_profile
        from dino_finetune.inference import OvenPostprocessor, load_gallery_bundle

        inp = self.meta["input"]
        transform = OvenTransforms(inp["img_dim"], inp["mean"], inp["std"], inp["img_interp"], is_train=False)
        classifier = self.classifier()
        np.testing.assert_array_equal(
            classifier._preprocess(self.image)[0], transform(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)).numpy(),
        )
        galleries, _ = load_gallery_bundle(self.bundle)
        profile = load_device_profile(self.bundle / "device_profile.json", max_rack=3, accessory_classes=["tray", "rack"])
        reference = OvenPostprocessor(
            calibration=self.calibration, profile=profile, container_classes=["bowl", "foil"],
            accessory_classes=["tray", "rack"], galleries=galleries,
        )
        rng = np.random.default_rng(45)
        outputs = {
            "is_oven_prob": rng.random(12).astype(np.float32),
            "food_prob": rng.random(12).astype(np.float32),
            "container_prob": rng.random((12, 2)).astype(np.float32),
            "accessory_prob": rng.random((12, 2)).astype(np.float32),
            "rack_raw": rng.normal(size=(12, 4)).astype(np.float32),
            "proj": rng.normal(size=(12, 2)).astype(np.float32),
            "cls": rng.normal(size=(12, 2)).astype(np.float32),
        }
        standalone = self.postprocessor()
        for device in (None, "A", "B"):
            with self.subTest(device_model=device):
                self.assertEqual(
                    standalone(outputs, device_model=device, with_scores=True),
                    reference(outputs, device_model=device, with_scores=True),
                )


if __name__ == "__main__":
    unittest.main()
