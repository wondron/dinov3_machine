from typing import Optional
import torch.nn.functional as F
import torch


class SegmentationMetricAccumulator:
    """在完整验证集上累计混淆矩阵并计算分割指标。"""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        mode: str = "argmax",
        threshold: float = 0.5,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.mode = str(mode).strip().lower()
        self.threshold = float(threshold)

        if self.num_classes <= 0:
            raise ValueError("num_classes 必须是大于 0 的整数")
        if self.mode not in {"argmax", "prob_threshold"}:
            raise ValueError("分割指标模式仅支持 argmax / prob_threshold")
        if self.mode == "prob_threshold" and self.num_classes != 2:
            raise ValueError("prob_threshold 仅支持二分类分割")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("分割概率阈值必须在 0 到 1 之间")

        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device="cpu",
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.ndim != 4 or targets.ndim != 3:
            raise ValueError(
                f"分割指标输入维度错误：logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
            )
        if int(logits.shape[1]) != self.num_classes:
            raise ValueError(
                f"分割 logits 类别数不一致：期望 {self.num_classes}，实际 {int(logits.shape[1])}"
            )

        if self.mode == "prob_threshold":
            foreground_prob = torch.softmax(logits.float(), dim=1)[:, 1]
            predictions = (foreground_prob >= self.threshold).long()
        else:
            predictions = torch.argmax(logits, dim=1)

        valid = torch.ones_like(targets, dtype=torch.bool)
        if self.ignore_index is not None:
            valid &= targets != self.ignore_index

        valid_targets = targets[valid].long()
        valid_predictions = predictions[valid].long()
        if valid_targets.numel() == 0:
            return

        min_target = int(valid_targets.min().item())
        max_target = int(valid_targets.max().item())
        if min_target < 0 or max_target >= self.num_classes:
            raise ValueError(
                f"验证 mask 标签越界：min={min_target} max={max_target} "
                f"num_classes={self.num_classes}"
            )

        flat_indices = valid_targets * self.num_classes + valid_predictions
        batch_confusion = torch.bincount(
            flat_indices,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += batch_confusion.cpu()

    def compute(self) -> dict[str, float]:
        confusion = self.confusion.to(dtype=torch.float64)
        intersection = torch.diag(confusion)
        target_count = confusion.sum(dim=1)
        prediction_count = confusion.sum(dim=0)
        union = target_count + prediction_count - intersection

        present = union > 0
        iou = torch.zeros_like(union)
        iou[present] = intersection[present] / union[present]

        miou = float(iou[present].mean().item()) if bool(present.any()) else 0.0
        foreground_present = present.clone()
        if foreground_present.numel() > 0:
            foreground_present[0] = False
        fg_iou = (
            float(iou[foreground_present].mean().item())
            if bool(foreground_present.any())
            else 0.0
        )

        total = float(confusion.sum().item())
        pixel_acc = float(intersection.sum().item()) / total if total > 0 else 0.0
        return {
            "miou": miou,
            "fg_iou": fg_iou,
            "pixel_acc": pixel_acc,
        }


def compute_iou_metric(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: int | None = None,
    eps: float = 1e-6,
    exclude_background: bool = False,
) -> float:
    num_classes = y_hat.shape[1]
    pred = torch.argmax(y_hat, dim=1)

    ious = []
    start_class = 1 if exclude_background and num_classes > 1 else 0
    for c in range(start_class, num_classes):
        pred_c = (pred == c)
        y_c = (y == c)

        if ignore_index is not None:
            valid = (y != ignore_index)
            pred_c = pred_c & valid
            y_c = y_c & valid

        intersection = (pred_c & y_c).sum().float()
        union = (pred_c | y_c).sum().float()

        if union > 0:
            ious.append((intersection + eps) / (union + eps))

    if len(ious) == 0:
        return pred.new_tensor(0.0).float()

    return torch.mean(torch.stack(ious))


def soft_dice_loss(
    logits: torch.Tensor,          # (B, C, H, W)
    targets: torch.Tensor,         # (B, H, W) long
    ignore_index: int = 255,
    eps: float = 1e-6,
    exclude_background: bool = True,
) -> torch.Tensor:
    B, C, H, W = logits.shape

    valid = (targets != ignore_index)
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    probs = torch.softmax(logits, dim=1)
    t = targets.clone()
    t[~valid] = 0
    one_hot = F.one_hot(t, num_classes=C).permute(0, 3, 1, 2).float()

    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    one_hot = one_hot * valid_f

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + eps) / (denom + eps)

    present = (one_hot.sum(dims) > 0)

    # 排除背景类
    if exclude_background:
        dice = dice[1:]  # class 0 作为背景
        present = present[1:]

    # 只对 batch 中出现的类别计算
    dice = dice[present] if present.sum() > 0 else dice

    return 1.0 - dice.mean()


def binary_soft_dice_loss_fg(
    logits: torch.Tensor,          # (B, 2, H, W)
    targets: torch.Tensor,         # (B, H, W) in {0,1,255}
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    # prob of foreground class=1
    probs_fg = torch.softmax(logits, dim=1)[:, 1]  # (B, H, W)

    valid = (targets != ignore_index)
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    tgt_fg = (targets == 1).float()

    probs_fg = probs_fg * valid.float()
    tgt_fg = tgt_fg * valid.float()

    inter = (probs_fg * tgt_fg).sum()
    denom = probs_fg.sum() + tgt_fg.sum()

    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice
