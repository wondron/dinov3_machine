from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """记录每轮指标，并保存 JSON 与训练曲线。"""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, float | int]] = []

    def update(self, epoch: int, metrics: Mapping[str, float | int]) -> None:
        epoch_number = int(epoch) + 1
        record: dict[str, float | int] = {"epoch": epoch_number}
        for name, value in metrics.items():
            record[str(name)] = int(value) if isinstance(value, int) else float(value)

        self.history = [item for item in self.history if int(item["epoch"]) != epoch_number]
        self.history.append(record)
        self.history.sort(key=lambda item: int(item["epoch"]))
        self.save()

    def state_dict(self) -> dict[str, Any]:
        return {"history": list(self.history)}

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        history = state.get("history", [])
        if not isinstance(history, list):
            raise ValueError("TrainingMonitor history 格式错误")
        self.history = [dict(item) for item in history if isinstance(item, dict)]

    def save(self) -> None:
        metrics_path = self.output_dir / "training_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(self.history, file, ensure_ascii=False, indent=2)

        try:
            self._save_curves()
        except Exception as exc:
            logger.warning("训练曲线保存失败，但指标 JSON 已保存：%s", exc)

    def _save_curves(self) -> None:
        if not self.history:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [int(item["epoch"]) for item in self.history]
        groups = [
            (
                "Loss",
                [
                    "train_loss",
                    "val_loss",
                    "train_seg_loss",
                    "val_seg_loss",
                    "train_cls_loss",
                    "val_cls_loss",
                    "train_ce",
                    "val_ce",
                    "train_dice",
                    "val_dice",
                ],
            ),
            (
                "Segmentation",
                ["val_miou", "val_iou", "val_fg_iou", "val_pixel_acc"],
            ),
            (
                "Classification",
                ["train_top1", "val_top1", "val_top5"],
            ),
            (
                "Score",
                ["score"],
            ),
        ]

        figure, axes = plt.subplots(2, 2, figsize=(14, 9))
        for axis, (title, names) in zip(axes.flat, groups):
            for name in names:
                values = [item.get(name) for item in self.history]
                if any(value is not None for value in values):
                    axis.plot(epochs, values, marker="o", label=name)
            axis.set_title(title)
            axis.set_xlabel("epoch")
            axis.grid(alpha=0.3)
            if axis.lines:
                axis.legend()

        figure.tight_layout()
        figure.savefig(self.output_dir / "training_curves.png", dpi=150)
        plt.close(figure)
