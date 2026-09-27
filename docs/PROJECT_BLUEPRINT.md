# 项目蓝图（长期维护版）

目标：作为训练 → PT 推理 → ONNX 导出 → ONNX 推理的唯一大纲，保证输入预处理、后处理、尺寸/插值、类别定义、ignore_index、阈值逻辑一致。

---

## 1) 仓库结构总览（按模块职责）

- `train_seg.py`：训练入口。构建编码器+LoRA+解码器，配置损失/优化器/调度器，训练与验证，保存权重与日志。
- `dino_finetune/`
  - `data.py`：数据集与预处理（Resize + Normalize），VOC/ADE/Binary 标签处理与 DataLoader。
  - `model/dino.py`：`DINOEncoderLoRA` 主模型（编码器 + LoRA + 解码器 + 上采样）。
  - `model/fpn_decoder.py` / `model/linear_decoder.py`：分割解码器实现。
  - `metrics.py`：IoU 与 Dice loss 计算。
  - `visualization.py`：可视化叠加工具。
- `script/`
  - `3-启动训练.sh`：训练启动样例（固定超参）。
  - `4-infer.py`：PT 推理（含阈值与 IoU 评估）。
  - `5-exportonnx.py`：ONNX 导出。
  - `6-inferOnnx.py`：ONNX 推理（含阈值与 IoU 评估）。
  - `7-compare_pt_onnx.py`：PT/ONNX 数值对齐检查脚本。
- `data/`：数据集/预训练权重目录（本地路径需自行配置）。
- `output/`、`infer_out/`、`logs/`：训练与推理产物。
- `third_party/`：外部依赖（DINOv2/v3 源码）。

---

## 2) 端到端数据流（含 shape/类型）

### 统一口径（必须一致）
- 输入归一化：`/255 + ImageNet mean/std`（训练与 PT/ONNX 推理一致）。
- 预测概率：使用 **softmax**（适配多分类），二分类取 `class=1` 作为前景概率。
- 阈值逻辑：`prob >= thr`。
- 尺寸策略：**推理尺寸必须与训练 IMG_DIM 完全一致**，不允许 letterbox。
- 插值策略：**图像 resize 用 `cv2.INTER_LINEAR`，mask/label 用 `cv2.INTER_NEAREST`**。
- 类别定义：Binary 固定 `0=背景, 1=前景`；多类按 `1..K` 递增。
- ignore_index：规范值为 `255`（当前数据未用到，但后续必须保留语义）。

### A. 训练数据流（`train_seg.py` + `dino_finetune/data.py`）
1. 读取图像  
   输入：`BGR uint8 (H0,W0,3)`  
   输出：同上（原始）
2. 转 RGB  
   输入：`BGR uint8 (H0,W0,3)`  
   输出：`RGB uint8 (H0,W0,3)`
3. 预处理 `SegTransforms`  
   - 图像：Resize → Normalize  
     输出：`torch.float32 (3,H,W)`  
   - mask：Resize(NEAREST) → Long  
     输出：`torch.int64 (H,W)`（Binary 为 `{0,1}`，未来允许 `{0,1,255}`）
4. DataLoader 组 batch  
   输出：  
   - `images: torch.float32 (B,3,H,W)`  
   - `masks: torch.int64 (B,H,W)`
5. 模型前向 `DINOEncoderLoRA`  
   - 编码器输出 patch 级特征  
   - 解码器输出 logits  
   输出：`logits: torch.float32 (B,C,H,W)`
6. Loss 与指标  
   - CE：`CrossEntropyLoss(ignore_index=255)`  
   - Dice：`soft_dice_loss / binary_soft_dice_loss_fg(ignore_index=255)`  
   - IoU：`argmax(logits)` + `ignore_index`
7. 保存  
   - LoRA + decoder 权重（`.pt`）  
   - 指标 JSON、日志

### B. PT 推理（`script/4-infer.py`）
1. 读取图像  
   输入：`BGR uint8 (H0,W0,3)`
2. 预处理（与训练一致）  
   输出：`torch.float32 (1,3,H,W)`
3. 模型推理  
   输出：`logits: torch.float32 (1,C,H,W)`
4. 后处理  
   - `prob = softmax(logits, dim=1)[:,1]` → `(H,W)`  
   - `prob_up = resize(prob, (H0,W0), INTER_LINEAR)`  
   - `mask = (prob_up >= thr) * 255` → `uint8 (H0,W0)`
5. IoU（可选）  
   - GT 读入 `int64 (H0,W0)`  
   - `pred_mask = (prob_up >= thr)`  
   - `IoU(pred_mask, gt, ignore_index)`

### C. ONNX 导出（`script/5-exportonnx.py`）
1. 构建编码器 + LoRA 模型  
   输入：与训练一致的 `IMG_DIM`、`n_classes`
2. 加载权重  
   - 预训练权重（encoder）  
   - LoRA + decoder 权重
3. 导出  
   输入：`dummy: torch.float32 (1,3,H,W)`  
   输出：`logits: (1,C,H,W)`  
   **需求：动态 batch（H/W 固定）**

### D. ONNX 推理（`script/6-inferOnnx.py`）
1. 预处理（禁止 letterbox）  
   输入：`BGR uint8 (H0,W0,3)`  
   输出：`float32 (1,3,H,W)`
2. ORT 推理  
   输出：`logits: float32 (1,C,H,W)`
3. 后处理（与 PT 一致）  
   - `prob = softmax(logits)[1]` → `(H,W)`  
   - `mask = (prob >= thr) * 255`  
   - resize 回原图：`prob INTER_LINEAR`，`mask INTER_NEAREST`
4. IoU（可选）  
   同 PT 推理流程

---

## 3) 问题清单（含影响程度、复现、文件、修复建议）

1. **P1 | ignore_index 未来可用性风险**  ?????2026-01-29?**
   - 复现：使用只包含 `{0,255}` 的标签，运行 `script/4-infer.py` 或 `script/6-inferOnnx.py`，`ignore_index=255`。  
   - 现象：`_read_label_as_index` 会把 `255` 自动转为 `1`，导致 ignore 被当成前景。  
   - 关联文件：`script/4-infer.py`、`script/6-inferOnnx.py`  
   - 建议：显式区分“255 为 ignore”的情况；当 `ignore_index=255` 时禁止自动 255→1 转换，或要求标签必须为 `{0,1,255}` 并校验。

2. **P1 | 评估指标与阈值口径不一致**  
   - 复现：设置 `thr != 0.5`，运行 PT/ONNX 推理并比较 IoU 与输出 mask。  
   - 现象：IoU 使用 `argmax(logits)`，而输出 mask 使用 `prob >= thr`，当 `thr != 0.5` 时结果不一致。  
   - 关联文件：`dino_finetune/metrics.py`、`script/4-infer.py`、`script/6-inferOnnx.py`  
   - 建议：评估时改为基于同一后处理（prob + thr）计算 IoU，或固定 `thr=0.5` 并写入配置。

3. **P1 | ONNX 导出缺少动态 batch**  
   - 复现：导出 ONNX 后尝试 batch>1 推理，维度不匹配。  
   - 关联文件：`script/5-exportonnx.py`  
   - 建议：`dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}`；保持 H/W 固定。

4. **P2 | 预处理插值口径在对齐脚本中不一致**  
   - 复现：`script/7-compare_pt_onnx.py` 对下采样使用 `INTER_AREA`，而训练/推理用 `INTER_LINEAR`。  
   - 影响：对齐脚本产生不必要的差异噪声。  
   - 关联文件：`script/7-compare_pt_onnx.py`  
   - 建议：统一为 `INTER_LINEAR`。

5. **P2 | Debug 可视化接口形状不匹配**  
   - 复现：`train_seg.py --debug`，`visualize_overlay` 期望 mask 形状 `(C,H,W)`，但传入 `(H,W)`。
   - 影响：可视化结果异常或潜在报错。  
   - 关联文件：`dino_finetune/visualization.py`、`train_seg.py`
   - 建议：让 `visualize_overlay` 同时支持 `(H,W)` 与 `(C,H,W)`，或调用前统一扩展维度。

6. **P2 | IMG_DIM / 路径配置分散，易造成链路不一致**  
   - 复现：训练与推理脚本分别使用不同 `IMG_DIM` 或硬编码路径。  
   - 影响：训练/推理不一致或环境迁移困难。  
   - 关联文件：`train_seg.py`、`script/4-infer.py`、`script/5-exportonnx.py`、`script/6-inferOnnx.py`
   - 建议：集中配置（YAML/JSON），统一读取。

7. **P2 | 推理脚本仅支持二分类后处理**  
   - 复现：将 `n_classes>2` 时使用现有推理脚本。  
   - 影响：prob/threshold/IoU 的逻辑不适配多类。  
   - 关联文件：`script/4-infer.py`、`script/6-inferOnnx.py`  
   - 建议：抽象多类后处理（argmax、可选阈值或 top-k）。

---

## 4) 重构路线图（M1/M2/M3 + 验收标准）

### M1 — 口径集中化（配置统一）
**目标**：将 IMG_DIM、mean/std、插值、阈值、ignore_index、类别表等集中管理。  
**验收标准**：  
- 训练、PT 推理、ONNX 推理均从同一配置文件读取关键参数。  
- 明确禁用 letterbox。  
- 预处理插值在三个链路完全一致（img=LINEAR, mask=NEAREST）。

### M2 — 指标与标签规范化
**目标**：让评估与输出 mask 逻辑一致，并规范 ignore_index。  
**验收标准**：  
- 当 `thr` 可配置时，IoU 与输出 mask 逻辑一致。  
- `ignore_index=255` 时 label 不会被自动改成前景。  
- 评估脚本对 `{0,1,255}` 能正确处理。

### M3 — 导出/部署对齐
**目标**：ONNX 支持动态 batch，PT/ONNX 数值对齐可复现。  
**验收标准**：  
- 导出 ONNX 支持 `batch=1..N`（H/W 固定）。  
- `7-compare_pt_onnx.py` 在统一插值后，logits 最大差异在可控阈值内。  
- 文档明确部署流程与检查项。

---

## 备注
本蓝图以 **当前约定口径** 为准：mean/std 启用、softmax 概率、`prob >= thr`、固定 IMG_DIM、禁止 letterbox、插值统一、binary 0/1、ignore_index=255 规范保留。
