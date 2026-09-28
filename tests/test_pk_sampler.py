from __future__ import annotations

from collections import Counter
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset

from dino_finetune.data import PKBatchSampler, build_train_loader
from dino_finetune.losses import SupConLoss


class IndexDataset(Dataset):
    def __init__(self, groups: list[int]) -> None:
        self.group_id = np.asarray(groups)

    def __len__(self) -> int:
        return len(self.group_id)

    def __getitem__(self, index: int) -> int:
        return index


class PKSamplerTests(unittest.TestCase):
    groups = [g for g in range(6) for _ in range(5)] + [-1] * 8

    def sampler(self, **overrides) -> PKBatchSampler:
        kwargs = dict(
            group_ids=self.groups, batch_size=16, p_groups=4, non_oven_ratio=0.25,
            num_batches=6, seed=17, require_supcon=True,
        )
        kwargs.update(overrides)
        return PKBatchSampler(**kwargs)

    def assert_valid_supcon_batches(self, sampler: PKBatchSampler, batch_size: int) -> None:
        for batch in sampler:
            self.assertEqual(len(batch), batch_size)
            counts = Counter(self.groups[index] for index in batch)
            self.assertEqual(counts.pop(-1, 0), sampler.n_non_oven)
            self.assertEqual(len(counts), sampler.p)
            self.assertGreaterEqual(len(counts), 2)
            self.assertGreaterEqual(min(counts.values()), 2)

    def test_default_batch_preserves_non_oven_quota_and_four_groups(self) -> None:
        with self.assertNoLogs("dino_finetune.data", level="WARNING"):
            sampler = self.sampler()
        self.assertEqual((sampler.p, sampler.n_oven, sampler.n_non_oven), (4, 12, 4))
        self.assert_valid_supcon_batches(sampler, 16)

    def test_small_batch_adjusts_quota_and_p_and_backpropagates_supcon(self) -> None:
        with self.assertLogs("dino_finetune.data", level="WARNING") as logs:
            sampler = self.sampler(batch_size=4)
        self.assertEqual(len(logs.output), 2)
        self.assertEqual((sampler.p, sampler.n_oven, sampler.n_non_oven), (2, 4, 0))
        self.assert_valid_supcon_batches(sampler, 4)
        batch = next(iter(sampler))
        labels = torch.tensor([self.groups[index] for index in batch])
        feats = torch.randn(4, 8, generator=torch.Generator().manual_seed(42), requires_grad=True)
        loss = SupConLoss()(feats, labels)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(feats.grad).all())
        self.assertGreater(float(feats.grad.abs().sum()), 0.0)

    def test_high_non_oven_ratio_reserves_two_pairs(self) -> None:
        with self.assertLogs("dino_finetune.data", level="WARNING"):
            sampler = self.sampler(batch_size=8, non_oven_ratio=0.9)
        self.assertEqual((sampler.p, sampler.n_oven, sampler.n_non_oven), (2, 4, 4))
        self.assert_valid_supcon_batches(sampler, 8)

    def test_uneven_quota_has_at_least_two_samples_per_group(self) -> None:
        with self.assertLogs("dino_finetune.data", level="WARNING"):
            sampler = self.sampler(batch_size=9)
        self.assertEqual((sampler.p, sampler.n_oven), (3, 7))
        self.assert_valid_supcon_batches(sampler, 9)

    def test_available_groups_limit_p(self) -> None:
        with self.assertLogs("dino_finetune.data", level="WARNING"):
            sampler = self.sampler(group_ids=[0, 0, 1, 1])
        self.assertEqual((sampler.p, sampler.n_oven, sampler.n_non_oven), (2, 16, 0))
        for batch in sampler:
            self.assertEqual(Counter([0, 0, 1, 1][index] for index in batch), {0: 8, 1: 8})

    def test_impossible_supcon_constraints_fail_early(self) -> None:
        cases = [
            ({"batch_size": 3}, "batch_size >= 4"),
            ({"p_groups": 1}, "p_groups >= 2"),
            ({"group_ids": [0, 0, -1]}, "cavity_group"),
            ({"group_ids": [-1, -1]}, "cavity_group"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                self.sampler(**overrides)

    def test_disabled_supcon_allows_single_group_or_non_oven_only(self) -> None:
        for groups in ([0, 0], [-1, -1]):
            with self.subTest(groups=groups):
                sampler = self.sampler(
                    group_ids=groups, batch_size=1, p_groups=1, require_supcon=False,
                )
                self.assertEqual(len(list(sampler)), 6)
                self.assertTrue(all(len(batch) == 1 for batch in sampler))

    def test_same_epoch_repeats_and_set_epoch_changes_batches(self) -> None:
        sampler = self.sampler()
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        sampler.set_epoch(0)
        self.assertEqual(first, list(sampler))

    def test_group_samples_cycle_before_repeating(self) -> None:
        sampler = self.sampler(
            group_ids=[0, 0, 0], batch_size=6, p_groups=1, require_supcon=False,
        )
        for batch in sampler:
            self.assertEqual(set(batch[:3]), {0, 1, 2})
            self.assertEqual(set(batch[3:]), {0, 1, 2})

    def loader(self, groups: list[int], **overrides):
        kwargs = dict(
            batch_size=4, sampler_cfg={"type": "pk", "p_groups": 4, "non_oven_ratio": 0.25},
            steps_per_epoch=3, num_workers=0, seed=17, pin_memory=False, require_supcon=True,
        )
        kwargs.update(overrides)
        return build_train_loader(IndexDataset(groups), **kwargs)

    def test_loader_applies_active_supcon_constraints(self) -> None:
        with self.assertLogs("dino_finetune.data", level="WARNING"):
            loader = self.loader(self.groups)
        self.assertEqual((loader.batch_sampler.p, loader.batch_sampler.n_oven), (2, 4))
        self.assertEqual([len(batch) for batch in loader], [4, 4, 4])
        with self.assertRaisesRegex(ValueError, "cavity_group"):
            self.loader([0, 0, -1])

    def test_loader_disabled_supcon_and_random_sampling_remain_supported(self) -> None:
        loader = self.loader([0, 0, -1], batch_size=1, require_supcon=False)
        self.assertEqual([len(batch) for batch in loader], [1, 1, 1])
        random_loader = self.loader([-1, -1], batch_size=1, sampler_cfg={"type": "random"})
        self.assertEqual(sorted(int(batch.item()) for batch in random_loader), [0, 1])


if __name__ == "__main__":
    unittest.main()
