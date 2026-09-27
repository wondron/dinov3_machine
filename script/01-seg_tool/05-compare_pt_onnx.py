# compare_pt_onnx.py
# 用途：严格对齐预处理，逐图对比 PT 与 ONNX 的 logits/prob/mask 差异，输出可视化定位问题
#
# 示例：
#   python script/compare_pt_onnx.py \
#     --image /data/xxx/test_images/test/0001.jpg \
#     --onnx  output/dino_0126_binary_640.onnx \
#     --pt_ckpt /data/xxx/output/260124/lora_fpn_binary/lora_fpn_binary_best.pt \
#     --img_dim 640 640 \
#     --thr 0.5 \
#     --outdir compare_out
#
# 目录跑（随机抽 N 张）：
#   python script/compare_pt_onnx.py \
#     --image_dir /data/xxx/test_images/test \
#     --pattern "*.jpg" \
#     --max_n 50 \
#     --onnx output/dino_0126_binary_640.onnx \
#     --pt_ckpt /data/xxx/output/260124/lora_fpn_binary/lora_fpn_binary_best.pt \
#     --img_dim 640 640 --thr 0.5 --outdir compare_out
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 把项目根目录加进去


import glob
import argparse
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np
import torch
import onnxruntime as ort
from dino_finetune.config import load_config, resolve_interp, default_config_path, get_dino_paths


# =========================
# 0) 通用配置（按你项目的默认）
# =========================
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# ImageNet Normalize（你新版 ONNX preprocess 已经是这个）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# =========================
# 1) 工具函数
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def 选择_resize插值(src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> int:
    """与多数部署一致：缩小用 AREA，放大用 LINEAR"""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    if dst_h < src_h or dst_w < src_w:
        return cv2.INTER_AREA
    return cv2.INTER_LINEAR

def 统计(name: str, arr: np.ndarray) -> None:
    arrf = arr.astype(np.float64)
    print(f"【统计】{name:<18} 形状={tuple(arr.shape)} 类型={arr.dtype} "
          f"最小={arrf.min():.6f} 最大={arrf.max():.6f} 均值={arrf.mean():.6f} 标准差={arrf.std():.6f}")

def 保存差异热力图(prob_diff: np.ndarray, out_path: str) -> None:
    """prob_diff 建议传 abs(prob_pt - prob_onnx)"""
    d = np.abs(prob_diff)
    vmax = float(np.percentile(d, 99.5)) if d.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    vis = np.clip(d / vmax * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imwrite(out_path, heat)

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def mask_iou(mask1_u8: np.ndarray, mask2_u8: np.ndarray) -> Tuple[float, int, int]:
    m1 = mask1_u8 > 0
    m2 = mask2_u8 > 0
    inter = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    iou = float(inter / (union + 1e-12))
    return iou, inter, union

def 接近阈值区域比例(prob_pt: np.ndarray, prob_ox: np.ndarray, thr: float, eps: float = 0.01) -> float:
    near = (np.abs(prob_pt - thr) < eps) | (np.abs(prob_ox - thr) < eps)
    return float(near.mean())


# =========================
# 2) 预处理 / 后处理（严格对齐你当前 pipeline）
# =========================
def preprocess_pipeline(
    img_bgr: np.ndarray,
    img_dim: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    img_interp: int,
    do_norm: bool = True,
) -> np.ndarray:
    """
    ???x float32 NCHW
      - BGR -> resize -> RGB -> /255 -> (??Normalize) -> CHW -> NCHW
    """
    H0, W0 = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (img_dim[1], img_dim[0]), interpolation=img_interp)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0

    if do_norm:
        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean_arr) / std_arr

    x = np.transpose(x, (2, 0, 1))          # CHW
    x = x[None, ...].astype(np.float32)     # NCHW
    return x

def postprocess_from_logits(
    logits_nchw: np.ndarray,  # (1,2,h,w)
    orig_hw: Tuple[int, int],
    thr: float,
    img_interp: int,
    thr_mode: str = "ge",     # "ge" 或 "gt"，用于与你 PT/ONNX 最终一致
) -> Dict[str, np.ndarray]:
    """
    与你最终对齐方案一致：
      prob = sigmoid(logit1 - logit0)
      mask = prob >= thr 或 prob > thr
    返回：
      prob_up: 原图空间 float32
      mask_u8: 原图空间 uint8(0/255)
    """
    if logits_nchw.ndim != 4 or logits_nchw.shape[1] != 2:
        raise ValueError(f"logits 形状不正确，期望(1,2,H,W)，实际={logits_nchw.shape}")

    H0, W0 = orig_hw
    logit_fg = logits_nchw[0, 1] - logits_nchw[0, 0]  # (h,w)
    prob = sigmoid_np(logit_fg).astype(np.float32)

    prob_up = cv2.resize(prob, (W0, H0), interpolation=img_interp)

    if thr_mode == "gt":
        mask = (prob_up > thr).astype(np.uint8) * 255
    else:
        mask = (prob_up >= thr).astype(np.uint8) * 255

    return {"prob_up": prob_up.astype(np.float32), "mask_u8": mask}


# =========================
# 3) 运行 ONNX / PT
# =========================
def get_providers(prefer_trt: bool = False) -> List[str]:
    avail = ort.get_available_providers()
    providers: List[str] = []
    if prefer_trt and "TensorrtExecutionProvider" in avail:
        providers.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers

def run_onnx(onnx_path: str, x_nchw: np.ndarray, prefer_trt: bool = False) -> np.ndarray:
    providers = get_providers(prefer_trt=prefer_trt)
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: x_nchw})[0]
    print("【ONNX】执行提供者：", sess.get_providers())
    print(f"【ONNX】输入：名称={inp_name} 形状={sess.get_inputs()[0].shape} 类型={sess.get_inputs()[0].type}")
    return out

@torch.no_grad()
def run_pt(model: torch.nn.Module, x_nchw: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(x_nchw)
    if next(model.parameters()).is_cuda:
        x = x.to(device)
    else:
        x = x.to("cpu")
    logits = model(x)
    return logits.detach().cpu().numpy()

def build_pt_model_from_your_code(
    pt_ckpt: str,
    img_dim: Tuple[int, int],
    device: str,
    dino_local_repo: str,
    weight_path: str,
    cpu_pt: bool = False,
):
    """
    直接复用你项目里的模型构建逻辑（你贴的那套），保证 PT 与训练一致。
    """
    from dino_finetune import DINOEncoderLoRA

    # ====== 固定为你训练时的设置 ======
    DINO_TYPE = "dinov3"
    SIZE = "large"
    IMG_DIM = img_dim
    RANK_R = 8
    N_CLASSES = 2
    USE_LORA = True
    USE_FPN = True

    # ====== ckpt 对齐加载逻辑（与你贴的一致）======
    def _pick_state_dict(ckpt):
        if not isinstance(ckpt, dict):
            raise TypeError(f"checkpoint type not dict: {type(ckpt)}")
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

    def build_encoder():
        patch_size = 16 if DINO_TYPE == "dinov3" else 14
        backbones = {
            "small": f"{DINO_TYPE}_vits{patch_size}",
            "base":  f"{DINO_TYPE}_vitb{patch_size}",
            "large": f"{DINO_TYPE}_vitl{patch_size}",
            "giant": f"{DINO_TYPE}_vitg{patch_size}",
        }
        try:
            encoder = torch.hub.load(
                repo_or_dir=dino_local_repo,
                model=backbones[SIZE],
                source="local",
                pretrained=False,
            ).to(device).eval()
        except TypeError:
            encoder = torch.hub.load(
                repo_or_dir=dino_local_repo,
                model=backbones[SIZE],
                source="local",
            ).to(device).eval()

        ckpt = torch.load(weight_path, map_location="cpu")
        ckpt_sd = _pick_state_dict(ckpt)
        _auto_align_and_load(encoder, ckpt_sd)

        for p in encoder.parameters():
            p.requires_grad = False
        return encoder

    print("【PT】开始构建模型（与训练对齐）…")
    encoder = build_encoder()
    emb_dim = encoder.num_features

    model = DINOEncoderLoRA(
        encoder=encoder,
        r=RANK_R,
        emb_dim=emb_dim,
        img_dim=IMG_DIM,
        n_classes=N_CLASSES,
        use_lora=USE_LORA,
        use_fpn=USE_FPN,
    ).to(device).eval()

    print(f"【PT】加载权重：{pt_ckpt}")
    model.load_parameters(pt_ckpt)
    model.eval()

    if cpu_pt:
        print("【PT】强制切到 CPU 推理（用于排除 CUDA 数值差异）")
        model = model.to("cpu").eval()

    print("【PT】模型就绪！")
    return model


# =========================
# 4) 单张对比
# =========================
def compare_one(
    image_path: str,
    onnx_path: str,
    pt_ckpt: str,
    img_dim: Tuple[int, int],
    thr: float,
    outdir: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    img_interp: int,
    dino_local_repo: str,
    weight_path: str,
    prefer_trt: bool = False,
    cpu_pt: bool = False,
    save_x_npy: bool = True,
    do_norm: bool = True,
    thr_mode: str = "ge",
) -> None:
    ensure_dir(outdir)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读图失败：{image_path}")
    H0, W0 = img.shape[:2]

    print("===============================================")
    print(f"【输入】图片：{image_path}")
    print(f"【输入】原图尺寸：({H0},{W0})  模型输入尺寸：{img_dim}  阈值：{thr}  阈值模式：{thr_mode}")
    print(f"【输入】是否 Normalize：{do_norm}")

    # 1) preprocess（严格统一）
    x = preprocess_pipeline(img, img_dim=img_dim, mean=mean, std=std, img_interp=img_interp, do_norm=do_norm)
    统计("x_input", x)
    if save_x_npy:
        np.save(os.path.join(outdir, "x_input.npy"), x)
        print("【输出】已保存 x_input.npy（用于复现实验）")

    # 2) run ONNX
    logits_onnx = run_onnx(onnx_path, x, prefer_trt=prefer_trt)
    统计("logits_onnx", logits_onnx)

    # 3) run PT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_pt_model_from_your_code(
        pt_ckpt=pt_ckpt,
        img_dim=img_dim,
        device=device,
        dino_local_repo=dino_local_repo,
        weight_path=weight_path,
        cpu_pt=cpu_pt,
    )
    logits_pt = run_pt(model, x, device=device)
    统计("logits_pt", logits_pt)

    # 4) logits diff
    if logits_pt.shape != logits_onnx.shape:
        print(f"【警告】logits 形状不一致：PT={logits_pt.shape}  ONNX={logits_onnx.shape}")
    diff = np.abs(logits_pt - logits_onnx)
    统计("logits_absdiff", diff)
    print(f"【差异】logits 最大绝对值={diff.max():.6f}  平均绝对值={diff.mean():.6f}")

    np.save(os.path.join(outdir, "logits_pt.npy"), logits_pt)
    np.save(os.path.join(outdir, "logits_onnx.npy"), logits_onnx)
    np.save(os.path.join(outdir, "logits_diff.npy"), diff)
    np.save(os.path.join(outdir, "logits_diff_ch0.npy"), diff[0, 0])
    np.save(os.path.join(outdir, "logits_diff_ch1.npy"), diff[0, 1])
    print("【输出】已保存 logits_pt/onnx/diff 以及通道差异 npy")

    # 5) postprocess（严格统一）
    out_pt = postprocess_from_logits(logits_pt, (H0, W0), thr=thr, img_interp=img_interp, thr_mode=thr_mode)
    out_ox = postprocess_from_logits(logits_onnx, (H0, W0), thr=thr, img_interp=img_interp, thr_mode=thr_mode)

    prob_pt = out_pt["prob_up"]
    prob_ox = out_ox["prob_up"]
    mask_pt = out_pt["mask_u8"]
    mask_ox = out_ox["mask_u8"]

    统计("prob_pt_up", prob_pt)
    统计("prob_ox_up", prob_ox)

    prob_diff = np.abs(prob_pt - prob_ox)
    统计("prob_absdiff", prob_diff)
    print(f"【差异】prob 最大绝对值={prob_diff.max():.6f}  平均绝对值={prob_diff.mean():.6f}")

    # 6) 保存可视化
    cv2.imwrite(os.path.join(outdir, "mask_pt.png"), mask_pt)
    cv2.imwrite(os.path.join(outdir, "mask_onnx.png"), mask_ox)
    xor = cv2.bitwise_xor(mask_pt, mask_ox)
    cv2.imwrite(os.path.join(outdir, "mask_xor.png"), xor)

    cv2.imwrite(os.path.join(outdir, "prob_pt_gray.png"), np.clip(prob_pt * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(outdir, "prob_onnx_gray.png"), np.clip(prob_ox * 255, 0, 255).astype(np.uint8))
    保存差异热力图(prob_diff, os.path.join(outdir, "prob_diff_heat.png"))

    print("【输出】已保存 mask_pt/mask_onnx/mask_xor，以及 prob 灰度图 + 差异热力图")

    # 7) 指标
    xor_ratio = float((xor > 0).mean())
    iou, inter, union = mask_iou(mask_pt, mask_ox)
    near_ratio = 接近阈值区域比例(prob_pt, prob_ox, thr, eps=0.01)

    print(f"【MASK】xor_ratio={xor_ratio:.6f}（差异像素占比）")
    print(f"【MASK】IoU={iou:.6f}  intersection={inter}  union={union}")
    print(f"【MASK】接近阈值区域占比(eps=0.01)={near_ratio:.6f}")
    print(f"【完成】输出目录：{outdir}")


# =========================
# 5) 主入口（支持单张 / 目录抽样）
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=default_config_path("seg"), help="Path to YAML config")
    ap.add_argument("--onnx", required=True, help="ONNX 模型路径")
    ap.add_argument("--pt_ckpt", required=True, help="PT 权重（LoRA+decoder best.pt）")

    ap.add_argument("--image", default=None, help="单张图片路径")
    ap.add_argument("--image_dir", default=None, help="图片目录（用于批量/抽样）")
    ap.add_argument("--pattern", default="*.jpg", help="目录模式匹配（默认 *.jpg）")
    ap.add_argument("--max_n", type=int, default=1, help="目录模式下最多抽取 N 张对比（默认 1）")

    ap.add_argument("--img_dim", type=int, nargs=2, default=(640, 640), help="模型输入尺寸：H W")
    ap.add_argument("--thr", type=float, default=0.5, help="前景阈值")
    ap.add_argument("--thr_mode", type=str, default="ge", choices=["ge", "gt"], help="阈值比较：ge(>=) 或 gt(>)，需与你最终统一")
    ap.add_argument("--outdir", default="compare_out", help="输出目录")
    ap.add_argument("--prefer_trt", action="store_true", help="ORT 优先 TensorRT EP（如可用）")
    ap.add_argument("--cpu_pt", action="store_true", help="强制 PT 用 CPU 推理")
    ap.add_argument("--no_norm", action="store_true", help="关闭 Normalize（如果你的 ONNX 输入不含 mean/std）")
    ap.add_argument("--save_x_npy", type=int, default=1, help="1=保存 x_input.npy")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dino_local_repo, weight_path = get_dino_paths(cfg)
    cfg_input = cfg["input"]
    cfg_model = cfg["model"]
    cfg_post = cfg["postprocess"]

    if cfg_post.get("mask_mode") != "prob_threshold" or cfg_post.get("iou_mode") != "prob_threshold":
        raise ValueError("mask_mode/iou_mode must both be 'prob_threshold' for this script")

    img_dim = tuple(cfg_input["img_dim"])
    mean = tuple(cfg_input["mean"])
    std = tuple(cfg_input["std"])
    img_interp = resolve_interp(cfg_input["img_interp"])
    thr = float(cfg_post["thr"])
    cfg_ignore = int(cfg_post["ignore_index"])
    n_classes = int(cfg_model["n_classes"])
    letterbox = bool(cfg_input.get("letterbox", False))

    if letterbox:
        raise ValueError("config input.letterbox must be false (letterbox disabled)")
    if tuple(args.img_dim) != img_dim:
        print(f"[config] ignore CLI img_dim={tuple(args.img_dim)}, use config img_dim={img_dim}")
    if abs(args.thr - thr) > 1e-9:
        print(f"[config] ignore CLI thr={args.thr}, use config thr={thr}")
    if args.no_norm:
        print("[config] ignore CLI --no_norm, using config mean/std")
    if n_classes != 2:
        raise ValueError(f"n_classes={n_classes} not supported by this script (binary only)")


    do_norm = True
    thr_mode = "ge"
    if args.thr_mode != "ge":
        print(f"[config] ignore CLI thr_mode={args.thr_mode}, use thr_mode=ge")

    # 组装待对比图片列表
    paths: List[str] = []
    if args.image:
        paths = [args.image]
    elif args.image_dir:
        paths = sorted(glob.glob(os.path.join(args.image_dir, args.pattern)))
        if not paths:
            raise FileNotFoundError(f"目录未匹配到图片：{os.path.join(args.image_dir, args.pattern)}")
        if args.max_n > 0:
            paths = paths[:args.max_n]
    else:
        raise ValueError("请提供 --image 或 --image_dir")

    ensure_dir(args.outdir)
    print("啵啵啵🔥开始对比！")
    print(f"【配置】ONNX={args.onnx}")
    print(f"【配置】PT权重={args.pt_ckpt}")
    print(f"【配置】img_dim={img_dim} thr={thr} thr_mode={thr_mode} do_norm={do_norm}")
    print(f"【配置】输出目录={args.outdir}")
    print(f"【配置】图片数量={len(paths)}（目录模式可能已截断到 max_n）")

    # 每张图建一个独立子目录，避免覆盖
    for idx, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        one_out = os.path.join(args.outdir, f"{idx:03d}_{stem}")
        ensure_dir(one_out)

        print("\n")
        print(f"🚀 啵啵啵🔥进度：{idx+1}/{len(paths)}  当前：{p}")
        compare_one(
            image_path=p,
            onnx_path=args.onnx,
            pt_ckpt=args.pt_ckpt,
            img_dim=img_dim,
            thr=thr,
            outdir=one_out,
            mean=mean,
            std=std,
            img_interp=img_interp,
            dino_local_repo=dino_local_repo,
            weight_path=weight_path,
            prefer_trt=args.prefer_trt,
            cpu_pt=args.cpu_pt,
            save_x_npy=bool(args.save_x_npy),
            do_norm=do_norm,
            thr_mode=thr_mode,
        )

    print("\n✅ 啵啵啵🔥全部对比完成！")


if __name__ == "__main__":
    main()
