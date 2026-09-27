import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 把项目根目录加进去

import argparse
from dino_finetune import DINOEncoderLoRA
from dino_finetune.config import load_config, default_config_path, get_dino_paths
from train_seg import _pick_state_dict, _auto_align_and_load

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path("seg"), help="Path to YAML config")
    parser.add_argument("--lora_ckpt", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dino_local_repo, weight_path = get_dino_paths(cfg)
    cfg_input = cfg["input"]
    cfg_model = cfg["model"]
    cfg_datas = cfg['dataset']
    cfg_train = cfg['trainparams']
    
    img_dim = tuple(cfg_input["img_dim"])
    n_classes = int(cfg_model["n_classes"])
    args.size = cfg_model["size"]
    args.r = cfg_train["rank_r"]
    
    dataset_type = cfg_datas["type"]
    model_name = f"dino_{dataset_type}_{img_dim[0]}.onnx"
    save_dir = os.path.dirname(os.path.dirname(args.lora_ckpt))
    args.onnx_path = os.path.join(save_dir, model_name)
    
    
    print(args)

    # ---- load backbone ----
    backbones = {
        "large": "dinov3_vitl16",
    }

    encoder = torch.hub.load(
        repo_or_dir=dino_local_repo,
        model=backbones[args.size],
        source="local",
        pretrained=False,
    )
    print("读取模型成功!")

    ckpt = torch.load(weight_path, map_location="cpu")
    ckpt_sd = _pick_state_dict(ckpt)
    _auto_align_and_load(encoder, ckpt_sd)
    print("读取预训练权重成功!")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ---- build full model ----
    model = DINOEncoderLoRA(
        encoder=encoder,
        r=args.r,
        emb_dim=encoder.num_features,
        img_dim=img_dim,
        n_classes=n_classes,
        use_lora=True,
        use_fpn=True,
    )

    model.load_parameters(args.lora_ckpt)
    model.eval()
    model.cuda()
    print("读取LoRA权重成功!")
    # ---- dummy input ----
    dummy = torch.randn(
        1, 3, img_dim[0], img_dim[1],
        device="cuda"
    )

    # ---- export ----
    torch.onnx.export(
        model,
        dummy,
        args.onnx_path,
        opset_version=18,
        input_names=["image"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamic_axes=None,  # 👈 先别开动态
    )

    print(f"[OK] exported to {args.onnx_path}")

if __name__ == "__main__":
    main()
