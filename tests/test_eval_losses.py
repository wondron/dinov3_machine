"""验证各任务 loss 的跨 batch 分母、总 loss 和选模顺序。"""
from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from dino_finetune.config import LOSS_TERMS
from dino_finetune.engine import collect_outputs
from dino_finetune.losses import ArcFaceLoss, MultiLabelLoss, MultiTaskLoss, SupConLoss
from dino_finetune.model.oven import build_valid_mask


class TableModel(nn.Module):
    def __init__(self, size: int = 8, food_losses: list[float] | None = None):
        super().__init__()
        rng = torch.Generator().manual_seed(42)
        for key, shape in {
            "is_oven": (size,), "food": (size,), "container": (size, 2),
            "accessory": (size, 2), "rack_raw": (size, 3), "proj": (size, 4), "cls": (size, 4),
        }.items():
            self.register_buffer(key, torch.randn(shape, generator=rng))
        if food_losses is not None:
            self.food.copy_(torch.log(torch.expm1(torch.tensor(food_losses))))

    def forward(self, image, rack_count, floor_usable):
        out = {key: value[image.long()] for key, value in self.named_buffers()}
        valid = build_valid_mask(rack_count, floor_usable, out["rack_raw"].shape[1])
        out["rack"] = out["rack_raw"].masked_fill(~valid, float("-inf"))
        return out


def rows_for(group_ids: list[int]):
    rows = []
    for i, group in enumerate(group_ids):
        rows.append({
            "image": torch.tensor(i), "index": torch.tensor(i),
            "is_oven": torch.tensor(float(group >= 0)), "group_id": torch.tensor(group),
            "food": torch.tensor(0.), "food_known": torch.tensor(i < 5),
            "container": torch.tensor([float(i % 2), float(i % 3 == 0)]),
            "container_known": torch.tensor(i in (0, 2, 4, 6)),
            "accessory": torch.tensor([float(i % 2 == 0), float(i % 3 == 1)]),
            "accessory_known": torch.tensor(i in (1, 4)),
            "rack_level": torch.tensor(i % 2), "rack_known": torch.tensor(i in (0, 2, 4, 7)),
            "rack_count": torch.tensor(1), "floor_usable": torch.tensor(True),
        })
    return rows


def criterion(metric=None, *, kind="bce", weights=None):
    return MultiTaskLoss(
        weights or {name: float(i + 1) / 3 for i, name in enumerate(LOSS_TERMS)},
        container_loss=MultiLabelLoss(torch.tensor([2., 0.5]), kind=kind),
        accessory_loss=MultiLabelLoss(torch.tensor([0.5, 2.]), kind=kind),
        metric_loss=metric if metric is not None else ArcFaceLoss(4, 2),
        rack_smoothing=0.1,
    )


def evaluate(model, rows, loss_fn, *, batch_size=4, batch_sampler=None):
    loader = DataLoader(rows, batch_sampler=batch_sampler) if batch_sampler is not None else DataLoader(rows, batch_size=batch_size)
    return collect_outputs(model, loader, torch.device("cpu"), False, loss_fn)


class EvaluationLossTests(unittest.TestCase):
    def test_all_additive_terms_match_full_dataset_with_masks_and_uneven_batches(self):
        rows = rows_for([0, 0, 1, 1, 1, -1, -1, -1])
        rows = [rows[i] for i in (0, 5, 6, 7, 1, 2, 3, 4)]
        model = TableModel()
        for kind in ("bce", "focal"):
            loss_fn = criterion(kind=kind)
            full = next(iter(DataLoader(rows, batch_size=len(rows))))
            total, terms = loss_fn(model(full["image"], full["rack_count"], full["floor_usable"]), full)
            for batch_size in (1, 3, 4, 8):
                with self.subTest(kind=kind, batch_size=batch_size):
                    arrays, losses = evaluate(model, rows, loss_fn, batch_size=batch_size)
                    self.assertEqual(len(arrays["index"]), len(rows))
                    for name, expected in terms.items():
                        self.assertAlmostEqual(losses[f"loss_{name}"], expected.item(), delta=1e-5)
                    self.assertAlmostEqual(losses["loss"], total.item(), delta=1e-5)

    def test_batch_partition_does_not_reverse_food_loss_ranking(self):
        rows = rows_for([-1] * 8)
        rows = [rows[i] for i in (0, 5, 6, 7, 1, 2, 3, 4)]
        loss_fn = criterion(weights={name: float(name == "food") for name in LOSS_TERMS})
        model_a = TableModel(food_losses=[3., .1, .1, .1, .1, .1, .1, .1])
        model_b = TableModel(food_losses=[1.] * 8)
        for batch_size in (1, 3, 4, 8):
            with self.subTest(batch_size=batch_size):
                a = evaluate(model_a, rows, loss_fn, batch_size=batch_size)[1]
                b = evaluate(model_b, rows, loss_fn, batch_size=batch_size)[1]
                self.assertAlmostEqual(a["loss"], .68, places=6)
                self.assertAlmostEqual(b["loss"], 1., places=6)
                self.assertLess(a["loss"], b["loss"])

    def test_tasks_without_labels_report_zero_and_do_not_dilute_other_tasks(self):
        rows = rows_for([-1] * 8)
        for row in rows:
            for key in ("food_known", "container_known", "accessory_known", "rack_known"):
                row[key] = torch.tensor(False)
        loss_fn = criterion(SupConLoss())
        losses = evaluate(TableModel(), rows, loss_fn, batch_size=3)[1]
        for name in LOSS_TERMS:
            if name != "is_oven":
                self.assertEqual(losses[f"loss_{name}"], 0.)
        self.assertAlmostEqual(losses["loss"], loss_fn.weights["is_oven"] * losses["loss_is_oven"])

    def test_supcon_uses_only_anchors_with_positive_and_negative_examples(self):
        groups = [0, 0, 1, 2, 0, 0, 0, 1, 1, 0, 0, 0, -1, -1, -1, -1]
        rows = rows_for(groups)
        model, loss_fn = TableModel(size=16), criterion(SupConLoss())
        batches = [list(range(4)), list(range(4, 9)), list(range(9, 12)), list(range(12, 16))]
        first = loss_fn.metric_loss(model.proj[:4], torch.tensor(groups[:4])).item()
        second = loss_fn.metric_loss(model.proj[4:9], torch.tensor(groups[4:9])).item()
        losses = evaluate(model, rows, loss_fn, batch_sampler=batches)[1]
        self.assertAlmostEqual(losses["loss_proj"], (first * 2 + second * 5) / 7, places=6)
        self.assertAlmostEqual(
            losses["loss"], sum(loss_fn.weights[name] * losses[f"loss_{name}"] for name in LOSS_TERMS),
        )

    def test_collecting_outputs_without_criterion_has_no_losses(self):
        arrays, losses = evaluate(TableModel(), rows_for([-1] * 8), None)
        self.assertEqual(len(arrays["index"]), 8)
        self.assertEqual(losses, {})


if __name__ == "__main__":
    unittest.main()
