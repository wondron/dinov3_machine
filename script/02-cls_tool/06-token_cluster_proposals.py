# 15-token_cluster_proposals.py
import argparse
import glob
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dino_finetune import DINOEncoderLoRA
from dino_finetune.config import (
    default_config_path,
    get_dino_paths,
    load_config,
    resolve_interp,
)
from dino_finetune.data import SegTransforms
from dino_finetune.model.dino_multitask import DINOEncoderLoRA_MultiTask
from dino_finetune.proposals import token_cluster_proposals
from dino_finetune.utils.ckpt_cls import build_encoder

logger = logging.getLogger("token_cluster_demo")


def list_images(in_dir: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths: list[str] = []
    for ext in exts:
        paths += glob.glob(os.path.join(in_dir, ext))
    paths.sort()
    return paths


def resolve_image_path(image_arg: str | None) -> str:
    if image_arg:
        p = Path(image_arg)
        if p.is_dir():
            imgs = list_images(str(p))
            if not imgs:
                raise FileNotFoundError(f"目录下没有可用图片：{p}")
            return imgs[0]
        if not p.is_file():
            raise FileNotFoundError(f"图片路径不存在：{p}")
        return str(p)

    default_dir = Path("test_images/test")
    if not default_dir.is_dir():
        raise FileNotFoundError("未传入 --image，且默认目录 test_images/test 不存在")
    imgs = list_images(str(default_dir))
    if not imgs:
        raise FileNotFoundError("未传入 --image，且 test_images/test 下没有图片")
    return imgs[0]


def pick_model_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"checkpoint 类型错误，期望 dict，实际为 {type(ckpt).__name__}")


def infer_num_classes_cls(cfg: dict[str, Any], sd: dict[str, torch.Tensor]) -> int:
    if "cls_head.weight" in sd and hasattr(sd["cls_head.weight"], "shape"):
        w = sd["cls_head.weight"]
        if len(w.shape) == 2 and int(w.shape[0]) > 0:
            return int(w.shape[0])

    model_cls_cfg = cfg.get("model_cls", {}) or {}
    num_classes_cls = int(model_cls_cfg.get("num_classes", 0))
    if num_classes_cls > 0:
        return num_classes_cls

    logger.warning("ckpt 中未找到 cls_head.weight，且 model_cls.num_classes<=0，回退为 1")
    return 1


def build_multitask_model(
    cfg: dict[str, Any],
    ckpt_path: str,
    device: torch.device,
) -> DINOEncoderLoRA_MultiTask:
    dino_local_repo, weight_path = get_dino_paths(cfg)
    encoder = build_encoder(cfg, device, dino_local_repo=dino_local_repo, weight_path=weight_path)

    seg_img_dim = tuple(cfg["input"]["img_dim"])
    if seg_img_dim != (640, 640):
        raise ValueError(f"当前脚本要求 input.img_dim 为 [640,640]，实际为 {seg_img_dim}")

    tp = cfg.get("trainparams", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    n_classes_seg = int(model_cfg["n_classes"])
    emb_dim = int(getattr(encoder, "num_features", 0) or 0)
    if emb_dim <= 0:
        raise RuntimeError("无法从 encoder 获取 num_features")

    seg_model = DINOEncoderLoRA(
        encoder=encoder,
        r=int(tp.get("rank_r", 8)),
        emb_dim=emb_dim,
        img_dim=seg_img_dim,
        n_classes=n_classes_seg,
        use_lora=bool(tp.get("use_lora", True)),
        use_fpn=bool(tp.get("use_fpn", True)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = pick_model_state_dict(ckpt)
    num_classes_cls = infer_num_classes_cls(cfg, sd)
    pool = str((cfg.get("model_cls", {}) or {}).get("pool", "cls_token"))

    model = DINOEncoderLoRA_MultiTask(
        seg_model=seg_model,
        num_classes_cls=num_classes_cls,
        pool=pool,
        emb_dim=emb_dim,
    ).to(device)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info("加载多任务权重完成：缺失=%d，多余=%d", len(missing), len(unexpected))
    if missing:
        logger.warning("缺失参数（前 20 个）：%s", missing[:20])
    if unexpected:
        logger.warning("多余参数（前 20 个）：%s", unexpected[:20])

    model.eval()
    return model


def save_visuals(
    img_bgr: np.ndarray,
    food_mask640: torch.Tensor,
    proposals640: list[torch.Tensor],
    save_root: str,
    base_name: str,
) -> tuple[str, list[str]]:
    mask_dir = os.path.join(save_root, "masks")
    ov_dir = os.path.join(save_root, "overlays")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(ov_dir, exist_ok=True)

    h0, w0 = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    rng = np.random.default_rng(2026)

    saved_masks: list[str] = []
    if proposals640:
        for idx, pm in enumerate(proposals640):
            pm_u8 = (pm.to(dtype=torch.uint8).cpu().numpy() * 255).astype(np.uint8)
            pm_u8_orig = cv2.resize(pm_u8, (w0, h0), interpolation=cv2.INTER_NEAREST)

            mask_path = os.path.join(mask_dir, f"{base_name}_p{idx:03d}.png")
            cv2.imwrite(mask_path, pm_u8_orig)
            saved_masks.append(mask_path)

            contours, _ = cv2.findContours(pm_u8_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = tuple(int(v) for v in rng.integers(0, 256, size=3))
            cv2.drawContours(overlay, contours, -1, color, 2)
    else:
        # 没有 proposal 时，至少可视化 food mask 边界，保证有输出图。
        food_u8 = (food_mask640.to(dtype=torch.uint8).cpu().numpy() * 255).astype(np.uint8)
        food_u8_orig = cv2.resize(food_u8, (w0, h0), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(food_u8_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        mask_path = os.path.join(mask_dir, f"{base_name}_food.png")
        cv2.imwrite(mask_path, food_u8_orig)
        saved_masks.append(mask_path)

    overlay_path = os.path.join(ov_dir, f"{base_name}.jpg")
    cv2.imwrite(overlay_path, overlay)
    return overlay_path, saved_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Token 聚类 proposals demo")
    parser.add_argument("--config", type=str, default=default_config_path("mul"), help="YAML 配置路径")
    parser.add_argument("--ckpt", type=str, required=True, help="多任务 checkpoint 路径")
    parser.add_argument("--image", type=str, default=None, help="输入图片路径（可选）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args.outdir = os.path.dirname(args.ckpt)
    logger.info("输出目录：%s", args.outdir)
    cfg = load_config(args.config)
    cfg_input = cfg["input"]
    img_dim = tuple(cfg_input["img_dim"])
    if img_dim != (640, 640):
        raise ValueError(f"当前脚本要求 input.img_dim 为 [640,640]，实际为 {img_dim}")

    mean = tuple(cfg_input["mean"])
    std = tuple(cfg_input["std"])
    img_interp = resolve_interp(cfg_input["img_interp"])
    msk_interp = resolve_interp(cfg_input["mask_interp"])

    image_path = resolve_image_path(args.image)
    logger.info("输入图片：%s", image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备：%s", device)

    model = build_multitask_model(cfg, args.ckpt, device=device)

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    tfm = SegTransforms(
        resize_hw=img_dim,
        mean=mean,
        std=std,
        img_interp=img_interp,
        msk_interp=msk_interp,
    )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x640 = tfm(img_rgb).unsqueeze(0).to(device)

    with torch.inference_mode():
        seg_logits = model.forward_seg(x640)
    if seg_logits.ndim != 4:
        raise ValueError(f"分割输出维度错误，期望 (B,C,H,W)，实际为 {tuple(seg_logits.shape)}")

    if int(seg_logits.shape[1]) == 2:
        food_mask = torch.argmax(seg_logits, dim=1) == 1
    else:
        logger.warning("分割类别数为 %d，当前按 argmax>0 作为 food mask", int(seg_logits.shape[1]))
        food_mask = torch.argmax(seg_logits, dim=1) > 0

    food_ratio = float(food_mask.float().mean().item())
    logger.info("food mask 面积占比：%.4f", food_ratio)

    proposals = token_cluster_proposals(
        model=model,
        x640=x640,
        food_mask640=food_mask,
        patch=16,
        k_min=1,
        k_max=4,
        kmeans_iters=15,
        min_area640=800,
    )
    proposals0 = proposals[0] if proposals else []
    logger.info("proposal 数量：%d", len(proposals0))

    date_tag = datetime.now().strftime("%y%m%d")
    save_root = os.path.join(args.outdir, date_tag)
    os.makedirs(save_root, exist_ok=True)

    base_name = Path(image_path).stem
    overlay_path, mask_paths = save_visuals(
        img_bgr=img_bgr,
        food_mask640=food_mask[0],
        proposals640=proposals0,
        save_root=save_root,
        base_name=base_name,
    )

    logger.info("已保存 overlay：%s", overlay_path)
    logger.info("已保存 masks 数量：%d", len(mask_paths))
    logger.info("输出目录：%s", save_root)


if __name__ == "__main__":
    main()
