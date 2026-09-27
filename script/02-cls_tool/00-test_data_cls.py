# tools/smoke_cls_loader.py
from __future__ import annotations

import logging
from dino_finetune.config import default_config_path, load_config
from dino_finetune.data_cls import get_cls_dataloader
from dino_finetune.logging import setup_logging


def main():
    # =============================
    # 1️⃣ 日志初始化
    # =============================
    logger = setup_logging(name="smoke_cls", level=logging.INFO)

    logger.info("🚀 开始执行 分类数据管线 smoke 测试")
    # =============================
    # 2️⃣ 加载并校验配置
    # =============================
    cfg = load_config(default_config_path("cls"))
    if "dataset_cls" not in cfg:
        raise RuntimeError("config 中未配置 dataset_cls")

    if "input_cls" not in cfg:
        raise RuntimeError("config 中未配置 input_cls")

    ds_cfg = cfg["dataset_cls"]
    input_cfg = cfg["input_cls"]

    dataroot = ds_cfg["root"]
    split = ds_cfg.get("valid_split", "train")

    batch_size = int(ds_cfg.get("smoke_batch_size", 8))
    num_workers = int(ds_cfg.get("smoke_num_workers", 4))

    logger.info( 
        "📦 分类数据集配置："
        f"dataroot={dataroot}, split={split}, "
        f"batch_size={batch_size}, num_workers={num_workers}, "
        f"img_dim={input_cfg.get('img_dim')}"
    )

    # =============================
    # 3️⃣ 构建 DataLoader
    # =============================
    logger.info("🔧 构建分类 DataLoader（local_folder）")

    dl = get_cls_dataloader(
        dataroot=dataroot,
        split=split,
        input_cfg=input_cfg,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        ds_cfg=ds_cfg,
        pin_memory=True,
    )

    logger.info("✅ DataLoader 构建完成")

    # =============================
    # 4️⃣ 拉取一个 batch
    # =============================
    logger.info("📤 拉取一个 batch 进行验证")

    imgs, labels, metas = next(iter(dl))

    # =============================
    # 5️⃣ 打印关键信息（验收点）
    # =============================
    logger.info(
        "🧪 Batch 基本信息："
        f"batch_size={imgs.shape[0]}, "
        f"image_shape={tuple(imgs.shape)}, "
        f"labels_shape={tuple(labels.shape)}"
    )

    logger.info( f"🏷️ label 范围：min={int(labels.min())}, max={int(labels.max())}" )

    # 随机展示 3 条 meta
    show_n = min(3, len(metas))
    logger.info(f"🔍 随机展示 {show_n} 条样本 meta：")

    for i in range(show_n):
        m = metas[i]
        logger.info(
            f"  image_id={m.get('image_id')} | "
            f"leaf_id={m.get('leaf_id')} | "
            f"class={m.get('class_name')} | "
            f"path={m.get('path_hierarchy')}"
        )

    logger.info("✅ 分类数据管线 smoke 测试通过")


if __name__ == "__main__":
    main()
