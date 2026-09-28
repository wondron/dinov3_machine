from __future__ import annotations

import unittest

import torch
from torch import nn

from dino_finetune.utils.ckpt import auto_align_and_load, pick_state_dict


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 2)
        self.register_buffer("position_scale", torch.zeros(2))


class BackboneCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyBackbone()
        self.initial = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.complete = {key: torch.full_like(value, 7) for key, value in self.initial.items()}

    def assert_state_equals(self, expected: dict[str, torch.Tensor]) -> None:
        for key, actual in self.model.state_dict().items():
            with self.subTest(key=key):
                torch.testing.assert_close(actual, expected[key], rtol=0, atol=0)

    def assert_rejected_without_mutation(self, checkpoint: dict, message: str) -> None:
        with self.assertRaisesRegex(RuntimeError, message):
            auto_align_and_load(self.model, checkpoint)
        self.assert_state_equals(self.initial)

    def test_complete_checkpoint_loads_parameters_and_buffer(self) -> None:
        self.assertEqual(auto_align_and_load(self.model, self.complete), ([], []))
        self.assert_state_equals(self.complete)

    def test_nested_prefixed_checkpoints_load(self) -> None:
        for prefix in ("module.", "backbone.", "teacher.backbone.", "student.backbone.module."):
            with self.subTest(prefix=prefix):
                prefixed = {prefix + key: value for key, value in self.complete.items()}
                checkpoint = {"teacher": {"state_dict": prefixed}}
                self.assertEqual(auto_align_and_load(self.model, pick_state_dict(checkpoint)), ([], []))
                self.assert_state_equals(self.complete)

    def test_unrelated_head_is_ignored_and_reported(self) -> None:
        checkpoint = dict(self.complete, **{"head.weight": torch.ones(4, 2)})
        with self.assertLogs("dino_finetune.utils.ckpt", level="WARNING"):
            self.assertEqual(auto_align_and_load(self.model, checkpoint), ([], ["head.weight"]))
        self.assert_state_equals(self.complete)

    def test_missing_parameter_rejected(self) -> None:
        checkpoint = dict(self.complete)
        del checkpoint["proj.bias"]
        self.assert_rejected_without_mutation(checkpoint, "proj.bias")

    def test_missing_persistent_buffer_rejected(self) -> None:
        checkpoint = dict(self.complete)
        del checkpoint["position_scale"]
        self.assert_rejected_without_mutation(checkpoint, "position_scale")

    def test_shape_mismatch_rejected_before_valid_tensors_are_copied(self) -> None:
        checkpoint = dict(self.complete)
        checkpoint["proj.bias"] = torch.ones(3)
        self.assert_rejected_without_mutation(checkpoint, "形状不匹配.*proj.bias|proj.bias.*形状不匹配")

    def test_non_tensor_rejected_before_valid_tensors_are_copied(self) -> None:
        checkpoint = dict(self.complete)
        checkpoint["proj.bias"] = [1, 2]
        self.assert_rejected_without_mutation(checkpoint, "proj.bias.*Tensor")

    def test_sparse_tensor_rejected_before_valid_tensors_are_copied(self) -> None:
        checkpoint = dict(self.complete)
        checkpoint["proj.bias"] = checkpoint["proj.bias"].to_sparse()
        self.assert_rejected_without_mutation(checkpoint, "proj.bias.*布局不匹配")

    def test_meta_tensor_rejected_before_valid_tensors_are_copied(self) -> None:
        checkpoint = dict(self.complete)
        checkpoint["proj.bias"] = torch.empty(2, device="meta")
        self.assert_rejected_without_mutation(checkpoint, "proj.bias.*meta Tensor")

    def test_no_matching_keys_rejected(self) -> None:
        self.assert_rejected_without_mutation({"other.weight": torch.ones(2)}, "无法匹配")

    def test_bfloat16_weights_can_load_into_float32_model(self) -> None:
        checkpoint = {key: value.to(torch.bfloat16) for key, value in self.complete.items()}
        self.assertEqual(auto_align_and_load(self.model, checkpoint), ([], []))
        self.assert_state_equals(self.complete)


if __name__ == "__main__":
    unittest.main()
