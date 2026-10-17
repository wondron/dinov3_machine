"""训练及留一验证不得覆盖已有结果，续训应保留 checkpoint 和指标历史。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch
import yaml

import train
from dino_finetune.config import load_config
from dino_finetune.utils.training_monitor import TrainingMonitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("loo_outputs_under_test", PROJECT_ROOT / "script" / "4-留一型号验证.py")
LOO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOO)


class _Dataset:
    labels = [object()]
    transform = SimpleNamespace(describe=lambda: "test")

    def __len__(self):
        return len(self.labels)


class TrainingOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-training-output-", dir=PROJECT_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = load_config()
        self.cfg["output"]["root"] = str(self.root / "runs")

    def test_same_second_runs_receive_distinct_directories_and_keep_existing_files(self):
        with patch.object(train, "datetime") as date:
            date.now.return_value.strftime.return_value = "261017_120000"
            first = train.prepare_run_dir(self.cfg)
            (first / "ckpt_best.pt").write_bytes(b"keep")
            second = train.prepare_run_dir(self.cfg)
            third = train.prepare_run_dir(self.cfg)
        self.assertEqual(first.name, "261017_120000")
        self.assertEqual(second.name, "261017_120000_01")
        self.assertEqual(third.name, "261017_120000_02")
        self.assertEqual((first / "ckpt_best.pt").read_bytes(), b"keep")

    def test_explicit_empty_directory_is_accepted(self):
        output = self.root / "empty"
        output.mkdir()
        self.assertEqual(train.prepare_run_dir(self.cfg, output_dir=output), output)

    def test_explicit_nonempty_directory_is_rejected_without_changing_it(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "ckpt_best.pt"
        marker.write_bytes(b"previous training")
        with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
            train.prepare_run_dir(self.cfg, output_dir=output)
        self.assertEqual(marker.read_bytes(), b"previous training")

    def test_resume_uses_original_directory_and_rejects_conflicting_output(self):
        resume = self._make_resume()
        before = self._snapshot(resume.parent)
        self.assertEqual(train.prepare_run_dir(self.cfg, resume_path=resume), resume.parent)
        with self.assertRaisesRegex(ValueError, "沿用"):
            train.prepare_run_dir(self.cfg, resume_path=resume, output_dir=self.root / "other")
        self.assertEqual(self._snapshot(resume.parent), before)

    def _make_resume(self):
        output = self.root / "resume"
        output.mkdir()
        for name in ("config.yaml", "device_profile.json", "training_metrics.json", "ckpt_best.pt", "train.log"):
            (output / name).write_text(f"original {name}", encoding="utf-8")
        cfg = copy.deepcopy(self.cfg)
        cfg["trainparams"]["epochs"] = 1
        cfg["eval"]["run_stage2"] = False
        cfg["eval"]["run_test"] = False
        cfg["loss"]["pos_weight"] = "none"
        checkpoint = {
            "model": {}, "cfg": cfg,
            "labels": {key: cfg["labels"][key] for key in ("container", "accessory")},
            "max_rack": cfg["model"]["max_rack"], "criterion": {},
            "optimizer": {}, "scheduler": {}, "epoch": 0,
            "global_step": 5, "best_score": 0.8, "best_val_loss": 0.3,
            "monitor": {"history": [{"epoch": 1, "train_loss": 0.4}]},
        }
        resume = output / "ckpt_last.pt"
        torch.save(checkpoint, resume)
        return resume

    @staticmethod
    def _snapshot(directory):
        return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}

    @staticmethod
    def _args(resume, device="cpu"):
        return argparse.Namespace(resume=str(resume), config=None, output_dir=None, device=device, init=None)

    def test_unavailable_cuda_leaves_every_resume_file_unchanged(self):
        resume = self._make_resume()
        before = self._snapshot(resume.parent)
        with patch.object(train, "parse_args", return_value=self._args(resume, "cuda")), \
             patch.object(train, "setup_logging"), \
             patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                train.main()
        self.assertEqual(self._snapshot(resume.parent), before)

    def test_profile_validation_failure_does_not_overwrite_resume_snapshots(self):
        resume = self._make_resume()
        before = self._snapshot(resume.parent)
        with patch.object(train, "parse_args", return_value=self._args(resume)), \
             patch.object(train, "setup_logging"), \
             patch.object(train, "load_device_profile", side_effect=ValueError("bad profile")):
            with self.assertRaisesRegex(ValueError, "bad profile"):
                train.main()
        self.assertEqual(self._snapshot(resume.parent), before)

    def test_completed_resume_restores_monitor_and_keeps_both_checkpoints(self):
        resume = self._make_resume()
        before = self._snapshot(resume.parent)
        monitor = TrainingMonitor(resume.parent)
        model = Mock()
        model.parameters.return_value = []
        model.head_parameters.return_value = {"proj": []}
        criterion = Mock()
        criterion.parameters.return_value = []
        criterion.to.return_value = criterion
        optimizer = Mock()
        patches = {
            "parse_args": {"return_value": self._args(resume)},
            "setup_logging": {},
            "load_device_profile": {"return_value": {}},
            "load_split": {"return_value": ([object()], [])},
            "log_label_summary": {}, "warn_data_coverage": {},
            "make_dataset": {"return_value": _Dataset()},
            "build_train_loader": {"return_value": [object()]},
            "build_eval_loader": {"return_value": [object()]},
            "select_gallery_indices": {"return_value": []},
            "build_model": {"return_value": model},
            "build_optimizer": {"return_value": optimizer},
            "build_scheduler": {"return_value": Mock()},
            "TrainingMonitor": {"return_value": monitor},
        }
        with ExitStack() as stack:
            for name, kwargs in patches.items():
                stack.enter_context(patch.object(train, name, **kwargs))
            stack.enter_context(patch.object(train.MultiTaskLoss, "from_config", return_value=criterion))
            train.main()
        self.assertEqual(monitor.history, [{"epoch": 1, "train_loss": 0.4}])
        optimizer.load_state_dict.assert_called_once_with({})
        after = self._snapshot(resume.parent)
        for name in ("ckpt_last.pt", "ckpt_best.pt", "training_metrics.json", "train.log"):
            self.assertEqual(after[name], before[name])

    def test_stage2_restores_best_criterion_with_best_model(self):
        resume = self._make_resume()
        checkpoint = torch.load(resume, weights_only=True)
        checkpoint["cfg"]["eval"]["run_stage2"] = True
        checkpoint["criterion"] = {"weight": torch.tensor([2.0])}
        torch.save(checkpoint, resume)
        best = {
            "model": {"best": torch.tensor([1.0])},
            "criterion": {"weight": torch.tensor([1.0])},
            "epoch": 0, "score": 0.8,
        }
        torch.save(best, resume.parent / "ckpt_best.pt")
        model = Mock(lora_blocks=0)
        model.parameters.return_value = []
        model.head_parameters.return_value = {"proj": []}
        model.gallery_fingerprint.return_value = "best-fingerprint"
        criterion = Mock()
        criterion.parameters.return_value = []
        criterion.to.return_value = criterion
        calibration = {"tau": {"proj": 0.5}, "rack_conf_threshold": 0.5, "sources": {}}
        patches = {
            "parse_args": {"return_value": self._args(resume)}, "setup_logging": {},
            "load_device_profile": {"return_value": {}},
            "load_split": {"return_value": ([object()], [])},
            "log_label_summary": {}, "warn_data_coverage": {},
            "make_dataset": {"return_value": _Dataset()},
            "build_train_loader": {"return_value": [object()]},
            "build_eval_loader": {"return_value": [object()]},
            "select_gallery_indices": {"return_value": []},
            "build_model": {"return_value": model},
            "build_optimizer": {"return_value": Mock()},
            "build_scheduler": {"return_value": Mock()},
            "TrainingMonitor": {"return_value": TrainingMonitor(resume.parent)},
            "collect_outputs": {"return_value": ({}, {})},
            "calibrate_thresholds": {"return_value": calibration},
            "evaluate_outputs": {"return_value": ({}, {})},
        }
        with ExitStack() as stack:
            for name, kwargs in patches.items():
                stack.enter_context(patch.object(train, name, **kwargs))
            stack.enter_context(patch.object(train.MultiTaskLoss, "from_config", return_value=criterion))
            train.main()
        self.assertEqual(criterion.load_state_dict.call_args_list, [call(checkpoint["criterion"]), call(best["criterion"])])
        self.assertEqual(model.load_trainable_state_dict.call_args_list, [call(checkpoint["model"]), call(best["model"])])


class LeaveOneOutOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-loo-output-", dir=PROJECT_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.fold = self.run / "loo" / "G5A"
        self.cfg = load_config()

    def _simulate_training(self, cmd, **kwargs):
        config = Path(cmd[cmd.index("--config") + 1])
        output = Path(cmd[cmd.index("--output_dir") + 1])
        self.assertTrue(config.is_file())
        self.assertNotEqual(config.parent, output)
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        self.assertEqual(cfg["data"]["exclude_groups"], ["G5A"])
        self.assertFalse(cfg["eval"]["run_stage2"])
        self.assertFalse(cfg["eval"]["run_test"])
        self.assertEqual(cfg["trainparams"]["epochs"], 2)
        train.prepare_run_dir(cfg, output_dir=output)
        (output / "ckpt_best.pt").write_bytes(b"new checkpoint")
        return subprocess.CompletedProcess(cmd, 0)

    def test_fold_config_survives_training_directory_preparation(self):
        with patch.object(LOO.subprocess, "run", side_effect=self._simulate_training):
            LOO.train_fold(self.cfg, self.run, self.fold, "G5A", 2, "cpu")
        self.assertEqual(json.loads((self.fold / "train_done.json").read_text()), {"group": "G5A"})
        self.assertEqual((self.fold / "ckpt_best.pt").read_bytes(), b"new checkpoint")

    def test_retrain_archives_old_fold_including_checkpoints_and_results(self):
        self.fold.mkdir(parents=True)
        for name in ("ckpt_best.pt", "training_metrics.json", "loo_fold.json"):
            (self.fold / name).write_bytes(b"old result")
        with patch.object(LOO.subprocess, "run", side_effect=self._simulate_training):
            LOO.train_fold(self.cfg, self.run, self.fold, "G5A", 2, "cpu")
        backups = list(self.fold.parent.glob("G5A.previous_*"))
        self.assertEqual(len(backups), 1)
        for name in ("ckpt_best.pt", "training_metrics.json", "loo_fold.json"):
            self.assertEqual((backups[0] / name).read_bytes(), b"old result")
        self.assertEqual((self.fold / "ckpt_best.pt").read_bytes(), b"new checkpoint")

    def test_failed_fold_training_preserves_old_results_and_does_not_mark_done(self):
        self.fold.mkdir(parents=True)
        (self.fold / "ckpt_best.pt").write_bytes(b"previous checkpoint")
        with patch.object(LOO.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "train")):
            with self.assertRaises(subprocess.CalledProcessError):
                LOO.train_fold(self.cfg, self.run, self.fold, "G5A", 2, "cpu")
        backups = list(self.fold.parent.glob("G5A.previous_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "ckpt_best.pt").read_bytes(), b"previous checkpoint")
        self.assertFalse((self.fold / "train_done.json").exists())


if __name__ == "__main__":
    unittest.main()
