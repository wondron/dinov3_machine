from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Loss", ("train_loss", "val_loss")),
    (
        "Train loss terms",
        (
            "train_loss_is_oven",
            "train_loss_proj",
            "train_loss_food",
            "train_loss_container",
            "train_loss_accessory",
            "train_loss_rack",
        ),
    ),
    (
        "Val metrics",
        (
            "val_is_oven_acc",
            "val_food_acc",
            "val_container_map",
            "val_accessory_map",
            "val_rack_acc",
            "val_rack_acc_pm1",
            "val_device_top1_proj",
            "val_device_top1_raw",
        ),
    ),
    (
        "Grad norm",
        (
            "train_gn_is_oven",
            "train_gn_proj",
            "train_gn_food",
            "train_gn_container",
            "train_gn_accessory",
            "train_gn_rack",
            "train_gn_backbone",
        ),
    ),
    ("Score", ("score",)),
    ("LR", ("lr",)),
)


class TrainingMonitor:
    """记录每轮指标，并保存 JSON 与训练曲线。"""

    def __init__(
        self,
        output_dir: str | Path,
        groups: Sequence[tuple[str, Sequence[str]]] = DEFAULT_GROUPS,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.groups = groups
        self.history: list[dict[str, float | int]] = []

    def update(self, epoch: int, metrics: Mapping[str, float | int | None]) -> None:
        epoch_number = int(epoch) + 1
        record: dict[str, float | int] = {"epoch": epoch_number}
        for name, value in metrics.items():
            if value is None:
                continue
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
        cols = 3
        rows = (len(self.groups) + cols - 1) // cols
        figure, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
        for axis, (title, names) in zip(axes.flat, self.groups):
            for name in names:
                values = [item.get(name) for item in self.history]
                if any(value is not None for value in values):
                    axis.plot(epochs, values, marker="o", markersize=3, label=name)
            axis.set_title(title)
            axis.set_xlabel("epoch")
            axis.grid(alpha=0.3)
            if axis.lines:
                axis.legend(fontsize=8)
        for axis in list(axes.flat)[len(self.groups):]:
            axis.axis("off")

        figure.tight_layout()
        figure.savefig(self.output_dir / "training_curves.png", dpi=120)
        plt.close(figure)
