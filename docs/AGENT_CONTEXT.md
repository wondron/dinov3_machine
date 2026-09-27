**项目一句话说明**
本仓库提供基于 DINO ViT encoder 的语义分割与图像分类微调/推理/导出流程，并新增多任务联合训练模型与训练入口；分割模型为 `DINOEncoderLoRA`（可选 LoRA + Linear/FPN decoder），分类模型为 `DINOForClassification`（可选 LoRA + pool + Linear head），多任务模型为 `DINOEncoderLoRA_MultiTask`（共享 encoder + 分割头 + 分类头），并配有 PT/ONNX 一致性对齐脚本。`dino_finetune/model/dino.py::DINOEncoderLoRA`, `dino_finetune/model/dino_cls.py::DINOForClassification`, `dino_finetune/model/dino_multitask.py::DINOEncoderLoRA_MultiTask`, `train_seg.py::finetune_dino`, `train_cls.py::main`, `train_mul.py::main`, `script/7-compare_pt_onnx.py::main`, `script/14-compare_pt_onnx_cls.py::main`

**仓库结构总览（tree 精简版）**
```text
.
├─ configs/
│  ├─ default_seg.yaml
│  ├─ default_cls.yaml
│  └─ default_mul.yaml
├─ dino_finetune/
│  ├─ config.py
│  ├─ data.py
│  ├─ data_cls.py
│  ├─ metrics.py
│  ├─ model/
│  │  ├─ dino.py
│  │  ├─ dino_cls.py
│  │  ├─ dino_multitask.py
│  │  ├─ fpn_decoder.py
│  │  ├─ linear_decoder.py
│  │  └─ lora.py
│  └─ utils/ckpt_cls.py
├─ train_seg.py
├─ train_cls.py
├─ train_mul.py
└─ script/
   ├─ 4-infer.py
   ├─ 5-exportonnx.py
   ├─ 6-inferOnnx.py
   ├─ 7-compare_pt_onnx.py
   ├─ 11-infer_cls_top5.py
   ├─ 12-export_onnx_cls.py
   ├─ 13-infer_onnx_cls_top5.py
   ├─ 14-compare_pt_onnx_cls.py
   └─ smoke_cls_forward.py
```
代码线索: `train_seg.py::finetune_dino`, `train_cls.py::main`, `train_mul.py::main`, `dino_finetune/config.py::load_config`, `dino_finetune/data.py::get_dataloader`, `dino_finetune/data_cls.py::FoodClsDatasetLocalFolder`, `dino_finetune/model/dino.py::DINOEncoderLoRA`, `dino_finetune/model/dino_cls.py::DINOForClassification`, `dino_finetune/model/dino_multitask.py::DINOEncoderLoRA_MultiTask`

**配置系统（YAML 的关键字段、校验逻辑、默认值）**
- 配置入口与强校验: `dino_finetune/config.py::load_config` 读取 YAML 并强制调用 `validate_config`。
- 必填模块按任务校验：分割需要 `input/dataset/model/postprocess/trainparams`，分类需要 `input_cls/dataset_cls/model/model_cls/trainparams`，多任务需要两组模块。`dino_finetune/config.py::validate_config`
- `input`/`input_cls` 校验: `img_dim` 必须为长度 2 的 list；`mean`/`std` 长度必须为 3；`img_interp` 必须能被 `resolve_interp` 解析（`nearest|linear|area`）；`input` 必须包含 `mask_interp`；`letterbox=true` 会直接报错。`dino_finetune/config.py::_validate_input`, `dino_finetune/config.py::resolve_interp`
- `dataset`/`dataset_cls` 校验: `root` 非空；`dataset.type` 必须在 `binary|voc|ade20k|multiclass` 中；`dataset_cls.type` 只能为 `local_folder`。`dino_finetune/config.py::_validate_dataset`
- `model`/`model_cls` 校验: `model.n_classes` 必须为 >0 的 int；`model_cls.emb_dim` >0；`model_cls.num_classes` >=0（允许 0 表示运行时从 dataloader 推断）；`model_cls.pool` 只能为 `cls_token|mean`。`dino_finetune/config.py::_validate_model_seg`, `dino_finetune/config.py::_validate_model_cls`
- `postprocess` 校验: `thr` 必须为数值；`ignore_index` 必须为 int；`mask_mode` 必须与 `iou_mode` 相同；当 `mask_mode=argmax` 时 `thr` 必须为 0.5。`dino_finetune/config.py::_validate_postprocess`
- `trainparams` 校验: `use_lora/use_fpn` 必须为 bool；当 `use_lora=true` 时 `rank_r` 必须为 >0 的 int。`dino_finetune/config.py::_validate_trainparams`
- 多任务训练会额外读取 `trainparams.epochs/batch_size_seg/batch_size_cls/steps_per_epoch/lr/weight_decay/min_lr/warmup_steps/use_amp/grad_clip/dice_weight/loss_cls_weight/label_smoothing/num_workers_cls/lr_decoder/lr_lora/lr_cls` 等字段；多任务配置会强校验两个 batch size。`train_mul.py::main`, `dino_finetune/config.py::validate_config`
- `model.dino_local_repo` / `model.weight_path` 用于指定 DINO 本地仓库与预训练权重路径，多处训练/推理脚本通过 `get_dino_paths` 读取。`dino_finetune/config.py::get_dino_paths`, `train_seg.py::main`, `train_cls.py::_build_encoder`, `train_mul.py::main`, `script/4-infer.py::__main__`, `script/5-exportonnx.py::main`, `script/7-compare_pt_onnx.py::main`, `script/11-infer_cls_top5.py::main`, `script/12-export_onnx_cls.py::main`, `script/14-compare_pt_onnx_cls.py::main`, `script/smoke_cls_forward.py::main`
- 默认配置按任务拆分为 `configs/default_seg.yaml`、`configs/default_cls.yaml`、`configs/default_mul.yaml`，各入口通过 `default_config_path("seg"|"cls"|"mul")` 选择对应文件。

**数据流（训练/验证数据从哪来、关键 transforms、label 映射规则）**
分割（Segmentation）: 训练/验证通过 `dino_finetune/data.py::get_dataloader` 构建，使用 `SegTransforms`（cv2 resize 到 `img_dim`，/255 + mean/std，mask 用 `INTER_NEAREST`；若 mask 为 3 通道则 `argmax` 转 index）。`VOC` 用色板转 index 或 one-hot（默认 index），`ADE20K` mask 值从 1..150 映射到 0..149，`Binary` 直接读取 0/1/255 灰度 mask。`dino_finetune/data.py::SegTransforms`, `dino_finetune/data.py::PascalVOCDataset._convert_to_segmentation_mask`, `dino_finetune/data.py::ADE20kDataset.__getitem__`, `dino_finetune/data.py::BinarySegDataset.__getitem__`, `dino_finetune/data.py::get_dataloader`
分类（Classification / local_folder）: `FoodClsDatasetLocalFolder` 读取 `root/<split>/<class_name>/*` 目录结构，按类别目录构建标签；训练前会校验 train/valid 类别名称与顺序一致，并保存 `leaf_id_map.json`。`dino_finetune/data_cls.py::FoodClsDatasetLocalFolder`, `dino_finetune/data_cls.py::get_cls_dataloader`, `train_cls.py::preflight_check_and_prepare_labels`
多任务（Seg+Cls）: `train_mul.py::main` 同时构建 seg/cls dataloader，并复用 `preflight_check_and_prepare_labels` 生成并注入 `leaf_id_map.json`，确保分类标签一致。`train_mul.py::main`, `train_cls.py::preflight_check_and_prepare_labels`

**模型结构（encoder / LoRA / decoder / head 的组合方式）**
Encoder: 训练脚本通过 `torch.hub.load` 从本地 DINO 仓库加载 ViT，并用 `_auto_align_and_load` 对齐预训练权重；encoder 参数全部冻结。`train_seg.py::_auto_align_and_load`, `train_seg.py::main`, `dino_finetune/utils/ckpt_cls.py::build_encoder`
LoRA: `LoRA` 包装 `block.attn.qkv`，对 Q/V 注入低秩增量（K 不变），并在 `DINOEncoderLoRA`/`DINOForClassification` 中按 block 注入。`dino_finetune/model/lora.py::LoRA`, `dino_finetune/model/dino.py::DINOEncoderLoRA`, `dino_finetune/model/dino_cls.py::DINOForClassification`
Decoder（分割）: `DINOEncoderLoRA` 按 `use_fpn` 选择 `FPNDecoder` 或 `LinearClassifier`；FPN 取 `encoder.get_intermediate_layers` 并多次上采样（nearest）；Linear 将 patch tokens reshape 后做 1x1 conv；最终 logits 用 bilinear 插值回输入分辨率。`dino_finetune/model/dino.py::DINOEncoderLoRA.forward`, `dino_finetune/model/fpn_decoder.py::FPNDecoder`, `dino_finetune/model/linear_decoder.py::LinearClassifier`
Head（分类）: `DINOForClassification` 从 `encoder.forward_features` 取 `x_norm_clstoken` 或 patch 均值，做 L2 归一化，再经线性层输出 logits；forward 返回 `(logits, embedding)`。`dino_finetune/model/dino_cls.py::DINOForClassification.forward`
多任务（Seg+Cls）: `DINOEncoderLoRA_MultiTask` 共享 `seg_model.encoder`，分割分支复用 `seg_model`，分类分支做 pooling + `cls_head` 并返回 `(seg_logits, cls_logits, embedding)`。`dino_finetune/model/dino_multitask.py::DINOEncoderLoRA_MultiTask`

**训练入口（分割训练、分类训练分别怎么跑：命令、参数、输出产物）**
分割训练入口在 `train_seg.py::main`，最小命令示例:
```bash
python train_seg.py --config configs/default_seg.yaml --exp_name exp_seg --dataset binary --epochs 50 --batch_size 8
```
关键参数与产物: `--exp_name` 控制日志和指标文件名前缀；`--dataset` 影响数据集选择与 `n_classes` 校验；训练输出位于 `output/01-segment/<YYMMDD>/`，只保存 `ckpt_last.pt`、`ckpt_best.pt` 两个模型文件，并保存指标 JSON、训练曲线和默认日志文件 `<exp_name>.log`（可用 `--log_file` 覆盖）。`train_seg.py::main`, `train_seg.py::finetune_dino`
分类训练入口在 `train_cls.py::main`，最小命令示例:
```bash
python train_cls.py --config configs/default_cls.yaml --epochs 5 --batch_size 32 --num_workers 4 --device auto
```
关键参数与产物: 训练输出目录为 `output/02-classify/<YYMMDD>/`，只保存 `ckpt_last.pt`、`ckpt_best.pt` 两个 checkpoint，并保存 `leaf_id_map.json`、`config.yaml`、指标 JSON 和训练曲线。`train_cls.py::main`, `train_cls.py::preflight_check_and_prepare_labels`
多任务训练入口在 `train_mul.py::main`，最小命令示例:
```bash
python train_mul.py --config configs/default_mul.yaml --device auto
```
关键参数与产物: 输出目录为 `output/03-multi/<YYMMDD>/`，只保存 `ckpt_last.pt`、`ckpt_best.pt` 两个 checkpoint，并保存类别映射、指标 JSON 和训练曲线，支持 `--resume` 续训。`train_mul.py::main`, `train_cls.py::preflight_check_and_prepare_labels`

**推理与后处理（输入预处理、输出含义、阈值/argmax 逻辑）**
分割 PT 推理: `script/4-infer.py::infer_image` 使用 `SegTransforms` 进行 resize/normalize，模型输出 logits `(1,C,h,w)`，对 class=1 做 softmax 得到前景概率并上采样到原图大小，阈值化输出 0/255 mask，并可叠加 overlay 与 IoU 统计。`script/4-infer.py::infer_image`, `dino_finetune/data.py::SegTransforms`
分割 ONNX 推理: `script/6-inferOnnx.py::preprocess` 与 `postprocess` 进行 BGR->RGB、/255、mean/std 归一化与 softmax 概率阈值化；如 `letterbox` 为真会反变换，但配置校验与脚本均要求 `letterbox=false`。`script/6-inferOnnx.py::preprocess`, `script/6-inferOnnx.py::postprocess`, `dino_finetune/config.py::_validate_input`
分类 PT 推理: `script/11-infer_cls_top5.py::_preprocess_one` 进行 resize/normalize，`DINOForClassification` 输出 `(logits, embedding)`，softmax 后取 Top-K，并用 `leaf_id_map.json` 反查类名。`script/11-infer_cls_top5.py::_preprocess_one`, `script/11-infer_cls_top5.py::_predict_topk`, `dino_finetune/model/dino_cls.py::DINOForClassification.forward`
分类 ONNX 推理: `script/13-infer_onnx_cls_top5.py::main` 通过 dataloader 取 batch，ONNX 输出 logits，softmax 后取 Top-K。`script/13-infer_onnx_cls_top5.py::main`
多任务推理: 未在代码中看到独立的多任务推理脚本（目前只有训练入口），需要补充。`train_mul.py::main`

**导出与一致性验证（PT vs ONNX 的对齐点、易错点清单）**
分割导出与验证: `script/5-exportonnx.py::main` 导出 segmentation ONNX（输出 logits），`script/6-inferOnnx.py::main` 做 ONNX 推理，`script/7-compare_pt_onnx.py::main` 对齐 PT/ONNX 的 preprocess 与 postprocess 并输出差异可视化。对齐关键点包括 `img_dim`、`mean/std`、`img_interp`、`thr` 与 `mask_mode=iou_mode=prob_threshold`。`script/5-exportonnx.py::main`, `script/6-inferOnnx.py::main`, `script/7-compare_pt_onnx.py::postprocess_from_logits`, `dino_finetune/config.py::_validate_postprocess`
分类导出与验证: `script/12-export_onnx_cls.py::main` 导出 logits+embedding 的 ONNX（batch 动态轴），`script/14-compare_pt_onnx_cls.py::main` 对齐 PT/ONNX 的 logits/embedding 并统计最大差与 cosine 相似度。`script/12-export_onnx_cls.py::main`, `script/14-compare_pt_onnx_cls.py::main`
多任务导出: 未在代码中看到多任务模型的导出与一致性验证脚本，需要补充。`dino_finetune/model/dino_multitask.py::DINOEncoderLoRA_MultiTask`

**常见坑与排查（至少 8 条，必须能关联到代码位置）**
1. `train_seg.py` 里 `--warmup_epochs` 只解析不使用，学习率 warmup 实际由 `--warmup_steps` 控制。`train_seg.py::main`, `train_seg.py::finetune_dino`
2. `script/3-启动训练.sh` 传入了 `train_seg.py` 未定义的参数（如 `--size/--img_dim/--r/--use_lora/--use_fpn`），会触发 “unrecognized arguments”。`script/3-启动训练.sh`, `train_seg.py::main`
3. 单任务入口应使用对应默认配置；若混用配置，训练脚本访问任务专属字段时会报错。`dino_finetune/config.py::default_config_path`, `dino_finetune/config.py::validate_config`
4. `input.letterbox=true` 会在配置校验阶段直接报错，ONNX 推理脚本也会拒绝 letterbox。`dino_finetune/config.py::_validate_input`, `script/6-inferOnnx.py::main`
5. 分割训练选择数据集类型依赖 `train_seg.py --dataset`，而不是 `dataset.type`（后者只参与校验）。`train_seg.py::main`, `dino_finetune/config.py::_validate_dataset`
6. `DINOEncoderLoRA` 要求 `img_dim` 能被 `encoder.patch_size` 整除，`train_seg.py` 只提示 warning，但实例化时仍会 assert 失败。`dino_finetune/model/dino.py::DINOEncoderLoRA.__init__`, `train_seg.py::main`
7. 三个训练入口分别写入 `output/01-segment`、`output/02-classify`、`output/03-multi` 下的日期目录。`train_seg.py::main`, `train_cls.py::main`, `train_mul.py::main`
8. 若未注入 `leaf_id_to_idx` 且未启用 `strict_label_map`，分类样本可能得到 `class_idx=-1`；`train_cls.py` 通过 preflight 注入并强制一致。`dino_finetune/data_cls.py::_map_leaf_to_idx`, `train_cls.py::preflight_check_and_prepare_labels`
9. 分割 PT 推理脚本使用 CLI `--ignore_index`，而不是 config 的 `postprocess.ignore_index`。`script/4-infer.py::main`
10. 分割 ONNX 导出脚本硬编码 DINO 路径并强制 `use_lora=True/use_fpn=True`，需要与训练设置人工对齐。`script/5-exportonnx.py::main`
11. 多任务训练依赖 `dataset_cls.train_split/valid_split`（默认 `train/valid`）；若数据集实际文件名不匹配会在 `_resolve_split_files` 处报错。`train_mul.py::main`, `dino_finetune/data_cls.py::_resolve_split_files`
12. 多任务训练会以 `encoder.num_features` 作为 `emb_dim`，若 `model_cls.emb_dim` 与之不一致会被覆盖并告警。`train_mul.py::main`

**待补充 / TODO（你认为当前仓库缺的文档或脚本）**
- 未在代码中看到多任务模型的推理与导出脚本，当前只有训练入口。`train_mul.py::main`, `dino_finetune/model/dino_multitask.py::DINOEncoderLoRA_MultiTask`
- 未在代码中看到统一的 PT/ONNX 推理入口脚本（配置完全对齐），当前分为 `script/4-infer.py` 与 `script/6-inferOnnx.py` 且部分超参硬编码。`script/4-infer.py::main`, `script/6-inferOnnx.py::main`
- 未在代码中看到对分类训练输出目录的可配置入口（如 CLI 或 config 字段），当前为硬编码路径。`train_cls.py::main`
- 未在代码中看到单独的数据格式文档，数据约束主要散落在 `data.py` 与 `data_cls.py` 中，建议补充独立说明。`dino_finetune/data.py::BinarySegDataset`, `dino_finetune/data_cls.py::FoodClsDatasetLocalFolder`
