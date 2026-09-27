import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # 把项目根目录加进去

import glob
import csv
import cv2
import numpy as np
import torch
from datetime import datetime
from typing import Tuple
from dino_finetune.data import SegTransforms
from dino_finetune.config import load_config, resolve_interp, default_config_path, get_dino_paths
from dino_finetune import DINOEncoderLoRA


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ====== 你训练里用过的 ckpt 对齐加载逻辑（用于 encoder 预训练权重）======
def _pick_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint 类型错误，期望 dict，实际为 {type(ckpt)}")

    tensor_like = 0
    for v in ckpt.values():
        if hasattr(v, "shape"):
            tensor_like += 1
    if tensor_like >= 50:
        return ckpt

    for k in ("model", "state_dict", "teacher", "student", "backbone", "net", "module"):
        v = ckpt.get(k, None)
        if isinstance(v, dict) and len(v) > 0:
            return _pick_state_dict(v)
    return ckpt


def _strip_prefix(sd, prefix: str):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _auto_align_and_load(model, ckpt_sd: dict):
    model_keys = set(model.state_dict().keys())
    candidates = [
        "", "module.", "model.",
        "backbone.",
        "teacher.", "student.",
        "teacher.backbone.", "student.backbone.",
        "teacher.model.", "student.model.",
        "teacher.module.", "student.module.",
        "teacher.backbone.module.", "student.backbone.module.",
    ]

    best_match = -1
    best_sd = None
    for p in candidates:
        sd_try = ckpt_sd if p == "" else _strip_prefix(ckpt_sd, p)
        if not sd_try:
            continue
        match = sum((k in model_keys) for k in sd_try.keys())
        if match > best_match:
            best_match = match
            best_sd = sd_try

    if best_sd is None or best_match == 0:
        msd = model.state_dict()
        filtered = {
            k: v for k, v in ckpt_sd.items()
            if k in msd and hasattr(v, "shape") and v.shape == msd[k].shape
        }
        if not filtered:
            raise RuntimeError("无法从 checkpoint 中匹配到任何模型参数（best_match=0）")
        best_sd = filtered

    model.load_state_dict(best_sd, strict=False)


def build_encoder(dino_local_repo: str, weight_path: str, dino_type: str, size: str, device: str):
    patch_size = 16 if dino_type == "dinov3" else 14
    backbones = {
        "small": f"{dino_type}_vits{patch_size}{'_reg' if dino_type == 'dinov2' else ''}",
        "base": f"{dino_type}_vitb{patch_size}{'_reg' if dino_type == 'dinov2' else ''}",
        "large": f"{dino_type}_vitl{patch_size}{'_reg' if dino_type == 'dinov2' else ''}",
        "giant": f"{dino_type}_vitg{patch_size}{'_reg' if dino_type == 'dinov2' else ''}",
        "huge": f"{dino_type}_vith{patch_size}{'plus' if dino_type == 'dinov3' else ''}{'_reg' if dino_type == 'dinov2' else ''}",
    }
    if size not in backbones:
        raise ValueError(f"不支持的模型尺寸：{size}，可选值：{sorted(backbones)}")

    # 1) 加载模型结构
    try:
        encoder = torch.hub.load(
            repo_or_dir=dino_local_repo,
            model=backbones[size],
            source="local",
            pretrained=False,
        ).to(device).eval()
    except TypeError:
        encoder = torch.hub.load(
            repo_or_dir=dino_local_repo,
            model=backbones[size],
            source="local",
        ).to(device).eval()

    # 2) 加载 DINO 预训练权重（与你训练一致）
    ckpt = torch.load(weight_path, map_location="cpu")
    ckpt_sd = _pick_state_dict(ckpt)
    _auto_align_and_load(encoder, ckpt_sd)

    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def build_model(
    pt_path: str,
    dino_local_repo: str,
    weight_path: str,
    dino_type: str,
    size: str,
    img_dim: Tuple[int, int],
    rank_r: int,
    n_classes: int,
    use_lora: bool,
    use_fpn: bool,
    device: str,
):
    encoder = build_encoder(dino_local_repo, weight_path, dino_type, size, device)
    emb_dim = encoder.num_features
    model = DINOEncoderLoRA(
        encoder=encoder,
        r=rank_r,
        emb_dim=emb_dim,
        img_dim=img_dim,
        n_classes=n_classes,
        use_lora=use_lora,
        use_fpn=use_fpn,
    ).to(device).eval()

    model.load_parameters(pt_path)
    model.eval()
    return model



def list_images(in_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(in_dir, e))
    paths.sort()
    return paths


def infer_logits(model, img_bgr, tfm: "SegTransforms", device=None) -> torch.Tensor:
    """返回 logits: (1,C,H,W)"""
    if device is None:
        device = next(model.parameters()).device
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = tfm(img_rgb).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
    return logits


def _resize_chw(chw: np.ndarray, size_wh: Tuple[int, int], interpolation: int) -> np.ndarray:
    return np.stack(
        [cv2.resize(chw[c], size_wh, interpolation=interpolation) for c in range(chw.shape[0])],
        axis=0,
    )


def infer_image(
    model,
    img_bgr,
    tfm: "SegTransforms",
    thr: float = 0.5,
    save_prob_npy: bool = False,
    mask_mode: str = "prob_threshold",
    n_classes: int = 2,
    img_interp: int = cv2.INTER_LINEAR,
    msk_interp: int = cv2.INTER_NEAREST,
    device=None,
):
    """返回：用于保存的 mask、类别 id mask、可选概率图、logits。"""
    H0, W0 = img_bgr.shape[:2]
    logits = infer_logits(model, img_bgr, tfm, device=device)  # (1,C,h,w)

    with torch.inference_mode():
        probs = torch.softmax(logits, dim=1)[0].float().cpu().numpy()  # (C,h,w)

    if mask_mode == "prob_threshold":
        if n_classes != 2:
            raise ValueError("prob_threshold 模式仅支持二分类分割；多类别分割请在配置中使用 mask_mode=argmax")
        fg_np_up = cv2.resize(probs[1], (W0, H0), interpolation=cv2.INTER_LINEAR)
        mask_index = (fg_np_up >= thr).astype(np.uint8)
        mask_save = mask_index * 255
        prob_save = fg_np_up if save_prob_npy else None
        return mask_save, mask_index, prob_save, logits

    if mask_mode == "argmax":
        pred_model = np.argmax(probs, axis=0).astype(np.uint8)
        mask_index = cv2.resize(pred_model, (W0, H0), interpolation=msk_interp).astype(np.uint8)
        mask_save = mask_index
        prob_save = _resize_chw(probs, (W0, H0), interpolation=img_interp) if save_prob_npy else None
        return mask_save, mask_index, prob_save, logits

    raise ValueError(f"不支持的 mask_mode：{mask_mode}，可选值：prob_threshold / argmax")


def _build_palette(n_classes: int) -> np.ndarray:
    palette = np.zeros((max(n_classes, 1), 3), dtype=np.uint8)
    if n_classes <= 1:
        return palette
    palette[1] = np.array([0, 0, 255], dtype=np.uint8)  # BGR 红色，保持二分类旧可视化风格
    for c in range(2, n_classes):
        hue = np.uint8([[[((c - 1) * 37) % 180, 210, 255]]])
        palette[c] = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)[0, 0]
    return palette


def _draw_legend_items(img_bgr: np.ndarray, class_ids: list[int], palette: np.ndarray) -> np.ndarray:
    if len(class_ids) == 0:
        return img_bgr

    out = img_bgr.copy()
    h, w = out.shape[:2]
    pad = 8
    square = 14
    text_gap = 5
    item_gap_x = 12
    row_gap = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    item_sizes = []
    for cls_id in class_ids:
        text = str(cls_id)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        item_sizes.append((cls_id, text, square + text_gap + tw, max(square, th + baseline)))

    content_max_w = max(1, w - 2 * pad)
    rows: list[list[tuple[int, str, int, int]]] = []
    cur_row: list[tuple[int, str, int, int]] = []
    cur_w = 0
    for item in item_sizes:
        add_w = item[2] if not cur_row else item_gap_x + item[2]
        if cur_row and cur_w + add_w > content_max_w:
            rows.append(cur_row)
            cur_row = [item]
            cur_w = item[2]
        else:
            cur_row.append(item)
            cur_w += add_w
    if cur_row:
        rows.append(cur_row)

    row_h = max(square, 18)
    panel_w = min(
        w,
        max(sum(item[2] for item in row) + item_gap_x * max(0, len(row) - 1) for row in rows) + 2 * pad,
    )
    panel_h = len(rows) * row_h + max(0, len(rows) - 1) * row_gap + 2 * pad
    panel_h = min(panel_h, h)

    panel = out.copy()
    cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (255, 255, 255), -1)
    cv2.addWeighted(panel, 0.72, out, 0.28, 0, dst=out)
    cv2.rectangle(out, (0, 0), (panel_w - 1, panel_h - 1), (30, 30, 30), 1)

    y = pad
    for row in rows:
        x = pad
        for cls_id, text, item_w, _ in row:
            color = tuple(int(v) for v in palette[cls_id].tolist())
            y1 = y + (row_h - square) // 2
            cv2.rectangle(out, (x, y1), (x + square, y1 + square), color, -1)
            cv2.rectangle(out, (x, y1), (x + square, y1 + square), (30, 30, 30), 1)
            cv2.putText(
                out,
                text,
                (x + square + text_gap, y + row_h - 4),
                font,
                font_scale,
                (20, 20, 20),
                thickness,
                cv2.LINE_AA,
            )
            x += item_w + item_gap_x
        y += row_h + row_gap

    return out


def overlay_mask(img_bgr, mask_index, n_classes: int, alpha=0.45):
    overlay = img_bgr.copy()
    palette = _build_palette(n_classes)
    present_class_ids: list[int] = []
    for cls_id in range(1, n_classes):
        m = (mask_index == cls_id)
        if not np.any(m):
            continue
        present_class_ids.append(cls_id)
        color = np.zeros_like(img_bgr)
        color[:, :] = palette[cls_id]
        overlay[m] = (overlay[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)
    overlay = _draw_legend_items(overlay, present_class_ids, palette)
    return overlay


def _build_label_map(label_dir: str):
    """按 base name 建索引：{basename: path}"""
    m = {}
    for p in list_images(label_dir):
        base = os.path.splitext(os.path.basename(p))[0]
        m[base] = p
    return m


def _read_label_as_index(label_path: str, ignore_index: int | None, n_classes: int) -> np.ndarray:
    """读取单通道类别 id 标注，允许包含 ignore_index。"""
    gt = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(f"标注不存在或无法读取：{label_path}")
    if gt.ndim == 3:
        if not np.all(gt == gt[:, :, :1]):
            raise ValueError(f"多类别标注必须是单通道类别 id 图：{label_path}")
        gt = gt[:, :, 0]

    gt = gt.astype(np.int64)
    valid = gt if ignore_index is None else gt[gt != ignore_index]
    if valid.size == 0:
        raise ValueError(f"标注没有有效像素：{label_path}")
    min_label = int(valid.min())
    max_label = int(valid.max())
    if min_label < 0 or max_label >= int(n_classes):
        unique = np.unique(gt).tolist()
        raise ValueError(
            f"标注值越界：{label_path}，期望有效标签范围为 0..{int(n_classes) - 1}，实际 unique={unique}"
        )
    return gt


def compute_iou_from_mask(
    pred_mask: np.ndarray,
    gt: np.ndarray,
    ignore_index: int | None = 255,
    n_classes: int = 2,
    eps: float = 1e-6,
    exclude_background: bool = False,
) -> float:
    pred = pred_mask.astype(np.int64)
    ious = []
    start_class = 1 if exclude_background and n_classes > 1 else 0
    for c in range(start_class, n_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)
        if ignore_index is not None:
            valid = (gt != ignore_index)
            pred_c = pred_c & valid
            gt_c = gt_c & valid

        intersection = np.logical_and(pred_c, gt_c).sum().astype(np.float32)
        union = np.logical_or(pred_c, gt_c).sum().astype(np.float32)
        if union > 0:
            ious.append((intersection + eps) / (union + eps))

    if len(ious) == 0:
        return 0.0
    return float(np.mean(np.stack(ious)))


def main(
    in_dir: str,
    out_dir: str,
    pt_path: str,
    dino_local_repo: str,
    weight_path: str,
    img_dim: Tuple[int, int],
    dino_type: str,
    size: str,
    rank_r: int,
    use_lora: bool,
    use_fpn: bool,
    thr: float = 0.5,
    save_prob_npy: int = 0,
    ignore_index: int | None = 255,
    n_classes: int = 2,
    mask_mode: str = "prob_threshold",
    iou_mode: str = "prob_threshold",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    img_interp: int = cv2.INTER_LINEAR,
    msk_interp: int = cv2.INTER_NEAREST,
):
    # === 自动加日期目录 ===
    date_tag = datetime.now().strftime("%y%m%d")  # 260125
    out_dir = os.path.join(out_dir, date_tag)

    os.makedirs(out_dir, exist_ok=True)
    mask_dir = os.path.join(out_dir, "masks")
    ov_dir = os.path.join(out_dir, "overlays")
    prob_dir = os.path.join(out_dir, "probs")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(ov_dir, exist_ok=True)
    if save_prob_npy:
        os.makedirs(prob_dir, exist_ok=True)

    paths = list_images(in_dir)
    if not paths:
        raise RuntimeError(f"输入目录未找到图片：{in_dir}")

    # === label dir: in_dir + "_label" ===
    label_dir = in_dir.rstrip("/\\") + "_label"
    has_label = os.path.isdir(label_dir)
    label_map = _build_label_map(label_dir) if has_label else {}

    if has_label:
        print(f"[标注] 找到标注目录：{label_dir}，标注数={len(label_map)}")
    else:
        print(f"[标注] 未找到标注目录：{label_dir}，跳过 IoU")

    if mask_mode != iou_mode:
        raise ValueError(f"mask_mode 与 iou_mode 必须一致，当前为 mask_mode={mask_mode}, iou_mode={iou_mode}")
    if n_classes > 2 and mask_mode != "argmax":
        raise ValueError("多类别分割模型必须使用 postprocess.mask_mode=argmax")

    model = build_model(
        pt_path,
        dino_local_repo,
        weight_path,
        dino_type=dino_type,
        size=size,
        img_dim=img_dim,
        rank_r=rank_r,
        n_classes=n_classes,
        use_lora=use_lora,
        use_fpn=use_fpn,
        device=DEVICE,
    )
    tfm = SegTransforms(
        resize_hw=img_dim,   # ✅ 用 config 里的 img_dim
        mean=mean,
        std=std,
        img_interp=img_interp,
        msk_interp=msk_interp,
    )

    # IoU 统计
    iou_list: list[float] = []
    csv_path = os.path.join(out_dir, "iou.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["image", "label", "miou"])

    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print("[跳过] 图片无法读取：", p)
            continue

        mask_save, mask_index, prob_up, logits = infer_image(
            model, img, tfm, thr,
            bool(save_prob_npy),
            mask_mode=mask_mode,
            n_classes=n_classes,
            img_interp=img_interp,
            msk_interp=msk_interp,
            device=DEVICE
        )

        base = os.path.splitext(os.path.basename(p))[0]

        # 保存 mask/overlay/prob
        mask_path = os.path.join(mask_dir, f"{base}.png")
        cv2.imwrite(mask_path, mask_save)

        ov = overlay_mask(img, mask_index, n_classes=n_classes)
        ov_path = os.path.join(ov_dir, f"{base}.jpg")
        cv2.imwrite(ov_path, ov)

        if save_prob_npy and prob_up is not None:
            np.save(os.path.join(prob_dir, f"{base}.npy"), prob_up.astype(np.float32))

        # 计算 IoU（若有 label）
        if has_label and (base in label_map):
            gt = _read_label_as_index(label_map[base], ignore_index=ignore_index, n_classes=n_classes)

            iou = compute_iou_from_mask(
                pred_mask=mask_index,
                gt=gt,
                ignore_index=ignore_index,
                n_classes=n_classes,
                exclude_background=(n_classes > 2),
            )
            iou_list.append(iou)
            p_base = os.path.basename(p)
            l_base = os.path.basename(label_map[base])
            writer.writerow([p_base, l_base, f"{iou:.6f}"])
        elif has_label:
            p_base = os.path.basename(p)
            writer.writerow([p_base, "", ""])

        if i % 10 == 0:
            print(f"[{i}/{len(paths)}] 已保存：{mask_path}")

    csv_f.close()

    if iou_list:
        miou = float(np.mean(iou_list))
        print(f"[IoU] 有标注图片数={len(iou_list)}，mIoU={miou:.6f}")
        print(f"[IoU] 单图结果已保存：{csv_path}")
    else:
        print("[IoU] 没有匹配到有效标注，跳过 mIoU。")

    print("推理完成，输出目录：", out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=default_config_path("seg"), help="YAML 配置文件路径")
    ap.add_argument("--pt_path", type=str, required=True, help="pt 文件路径")
    ap.add_argument("--in_dir", type=str, required=True, help="输入图片目录")
    ap.add_argument("--out_dir", type=str, default="z_infer_res/1_pt_out/01-segment", help="输出根目录")
    ap.add_argument("--save_prob_npy", type=int, default=0, help="1 表示保存概率 npy；多类别时保存 CxHxW 概率")
    ap.add_argument("--ignore_index", type=int, default=None, help="覆盖 config.postprocess.ignore_index；设为 -1 表示禁用")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dino_local_repo, weight_path = get_dino_paths(cfg)
    cfg_input = cfg["input"]
    cfg_model = cfg["model"]
    cfg_post = cfg["postprocess"]
    os.makedirs(args.out_dir, exist_ok=True)

    img_dim = tuple(cfg_input["img_dim"])
    mean = tuple(cfg_input["mean"])
    std = tuple(cfg_input["std"])
    img_interp = resolve_interp(cfg_input["img_interp"])
    msk_interp = resolve_interp(cfg_input["mask_interp"])
    
    cfg_train = cfg["trainparams"]
    thr = float(cfg_post["thr"])
    ignore_index = int(cfg_post["ignore_index"]) if args.ignore_index is None else int(args.ignore_index)
    if ignore_index < 0:
        ignore_index = None
    n_classes = int(cfg_model["n_classes"])
    mask_mode = str(cfg_post.get("mask_mode", "prob_threshold")).lower().strip()
    iou_mode = str(cfg_post.get("iou_mode", mask_mode)).lower().strip()

    main(
        args.in_dir,
        args.out_dir,
        args.pt_path,
        dino_local_repo,
        weight_path,
        img_dim=img_dim,
        dino_type=str(cfg_model["dino_type"]),
        size=str(cfg_model["size"]),
        rank_r=int(cfg_train["rank_r"]),
        use_lora=bool(cfg_train["use_lora"]),
        use_fpn=bool(cfg_train["use_fpn"]),
        thr=thr,
        save_prob_npy=args.save_prob_npy,
        ignore_index=ignore_index,
        n_classes=n_classes,
        mask_mode=mask_mode,
        iou_mode=iou_mode,
        mean=mean,
        std=std,
        img_interp=img_interp,
        msk_interp=msk_interp,
    )
