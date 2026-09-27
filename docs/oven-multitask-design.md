# 一体机多任务视觉识别模型 · 整体方案

> 版本 v2 · 2026-09-27
> 相对 v1 的改动：附件在一体机内外都参与训练；层位改为"0 = 底板层，1～N = 导轨层"的单一分类，不再单独设 rack_exist。

## 1. 目标

输入一张图片，输出：

| 输出 | 类型 |
|---|---|
| 是否为一体机内部 | 二分类 |
| 设备型号 | 检索式识别；库中没有的型号输出"未知型号" |
| 是否有食物 | 二分类 |
| 容器类型 | 多标签，10 类 |
| 附件类型 | 多标签，9 类；一体机内外都可能出现 |
| 层位 | 分类：0 = 底板层，1～N = 导轨层；只在一体机内部输出 |

**设计原则**

- 设备结构知识（层数、能否直接放底板、支持的附件、内腔分组）写在设备配置表里，模型只负责视觉判断。
- 型号采用检索式识别。新增型号只需补充参考图和配置，不改网络结构。
- 所有带 null 的标签都用 loss 掩码处理，不拿占位值去参与训练。

---

## 2. 整体架构

```
Image
 └─ DINOv3（冻结）→ CLS token + patch tokens
     ├─ Is-Oven Head ─────────── 是否一体机内部（二分类）
     ├─ Food Head ────────────── 是否有食物（二分类）
     ├─ Container Head ───────── 容器（多标签 ×10）
     ├─ Accessory Head ───────── 附件（多标签 ×9；型号已知时屏蔽不支持的附件）
     ├─ Proj Head → 检索特征库 ── 设备型号 / 未知型号
     │                              │
     │                        Device Profile
     │           （rack_count、floor_usable、支持的附件、内腔分组）
     │                              │
     └─ Rack Head ────────────── 层位（0 底板层 + 1..N 导轨层，按 rack_count 掩码）
```

除 Proj Head 外，每个头都有独立的 attention pooling（见 4.1）。

---

## 3. 推理流程

1. **提特征**：DINOv3 输出 CLS token 和 patch tokens。
2. **判断是否一体机内部**：Is-Oven 判为"否"时，device = "无"，层位 = null。食物、容器、附件照常输出，附件不做设备屏蔽。
3. **识别型号**：用 Proj Head 的特征检索特征库。
   - top-1 相似度 < `tau`：输出"未知型号"，层位 = null，附件不做屏蔽，图片放入待补库池。
   - 否则输出型号和置信度。
4. **查配置表**：从 Device Profile 取出 `rack_count`、`floor_usable` 和支持的附件列表。
5. **附件**：型号已知时，先屏蔽该型号不支持的附件，再按每类各自的阈值取正类；所有类都低于阈值时输出空列表。
6. **层位**：
   - 腔内没有任何物品（无食物、无容器、无附件）时，层位 = null。
   - 否则在有效类别中取 argmax：0 表示底板层，k 表示第 k 层导轨。
   - 最大概率低于阈值时，标记为低置信度。
   - 一致性检查：层位 ≥ 1 但附件为空时，同样标记为低置信度。
7. **食物、容器**：按各自的阈值输出。

**输出示例**

```json
{
  "is_oven": true,
  "device_model": "C9277A",
  "device_score": 0.93,
  "food_exist": true,
  "container_type": ["塑料容器"],
  "accessory_type": ["小炸篮"],
  "rack_level": 3,
  "rack_level_score": 0.88,
  "low_confidence": false
}
```

---

## 4. 模块设计

### 4.1 骨干与特征

- 骨干用 DINOv3 ViT-L（1024 维），第一阶段冻结。
- 同时保留 CLS token 和 patch tokens。
- 每个头配一个独立的 attention pooling：一个可学习的 query 对 patch tokens 做 cross-attention，输出和 CLS 拼接后送入各自的 MLP。这样层位头能自己关注导轨区域，附件头和容器头能关注小物体。
- 效果不够时，再解冻最后几个 block 或加 LoRA。**骨干一旦变动，特征库必须全部重建。**

```python
class AttnPool(nn.Module):
    def __init__(self, dim=1024, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, cls, patches):                    # cls: [B, D]  patches: [B, N, D]
        out, _ = self.attn(self.q.expand(len(patches), -1, -1), patches, patches)
        return torch.cat([cls, out.squeeze(1)], -1)     # [B, 2D]
```

### 4.2 Is-Oven Head

- 二分类，BCE，全部样本参与。
- 取代原 Device Head 里的"无"类。非一体机场景种类太杂，形不成聚类，不适合放进特征库。

### 4.3 设备识别（检索式）

**Proj Head**

- 结构：MLP 1024 → 512 → 256，输出做 L2 归一化。
- 只用一体机样本训练，loss 用 SupCon 或 ArcFace。
- 采用 PK 采样：每个 batch 取 P 个型号，每个型号取 K 张图。
- 数据增强让食物、附件、亮度和角度充分变化，迫使特征忽略这些因素。
- 训练集中的型号越多越好，停产机型和竞品也可以加进来，这样对新型号的泛化更好。
- 可选：池化时降低画面中心区域的权重，减少食物对特征的干扰。
- Proj 特征和原始 DINOv3 特征要做对比，通过留一型号验证（见第 9 节）决定最终用哪个。

**特征库**

- 每个型号准备 50～200 张参考图，覆盖空腔、放附件、放食物、灯亮/灯暗、不同层位等情况。
- 保留全部参考图的特征，推理时做 kNN（k = 10），按相似度加权投票。
- top-1 相似度 < `tau` 时判为"未知型号"。`tau` 用留一验证中"库外型号"的图片来标定。
- 内腔完全相同的型号，在配置表中归入同一个 `cavity_group`，检索结果以 cavity_group 为准。
- 特征库与 Proj Head、骨干的权重绑定版本，任何一方变动都要全部重建。

```python
class DeviceGallery:
    def __init__(self):
        self.feats, self.labels = [], []                   # 参考图特征 / 对应型号

    @torch.no_grad()
    def add(self, name, feats):                            # feats: [N, D]
        self.feats.append(F.normalize(feats, dim=-1))
        self.labels += [name] * len(feats)

    @torch.no_grad()
    def query(self, q, k=10, tau=0.5):                     # q: [D]
        bank = torch.cat(self.feats)
        top_s, top_i = (bank @ F.normalize(q, dim=-1)).topk(min(k, len(bank)))
        if top_s[0] < tau:
            return "未知型号", top_s[0].item()
        votes = {}
        for s, i in zip(top_s.tolist(), top_i.tolist()):
            votes[self.labels[i]] = votes.get(self.labels[i], 0) + s
        best = max(votes, key=votes.get)
        return best, votes[best] / sum(votes.values())     # 型号, 置信度

    def prototype(self, name):                             # 型号原型向量，供 Rack Head 条件实验使用
        idx = [i for i, n in enumerate(self.labels) if n == name]
        return F.normalize(torch.cat(self.feats)[idx].mean(0), dim=-1)
```

### 4.4 Device Profile

```json
{
  "C9277A":    {"rack_count": 5, "floor_usable": true, "cavity_group": "G5A", "accessories": ["烤盘", "小炸篮"]},
  "CQ09-i9":   {"rack_count": 3, "floor_usable": true, "cavity_group": "G3A", "accessories": []},
  "C87-i7Pro": {"rack_count": 5, "floor_usable": true, "cavity_group": "G5B", "accessories": []},
  "DB677":     {"rack_count": 2, "floor_usable": true, "cavity_group": "G2A", "accessories": []}
}
```

`floor_usable`、`cavity_group` 和 `accessories` 目前都是占位值，需要按实际机型填写。

### 4.5 Rack Head（层位）

**标签定义**

| 场景 | 层位标签 |
|---|---|
| 物品直接放在底板上（不用附件，或附件平放在底板上） | 0（底板层） |
| 主要附件插在第 k 层导轨上 | k（1～rack_count，从下往上数） |
| 空腔：没有食物、容器，也没有附件 | null |
| 非一体机 | null |

每张有物品的一体机内部图片都必须标出确定的层位，不设"无法判断"选项。

**结构与掩码**

- 输出维度为 `[B, MAX_RACK + 1]`，`MAX_RACK = 8`，覆盖现有及规划中机型的最大层数。
- 大于 `rack_count` 的类别屏蔽掉。第 0 类默认有效；`floor_usable = false` 的机型也要把第 0 类屏蔽。
- 训练目标就是层号本身，取值 0～rack_count。
- 基线版本不加型号条件，只靠层数掩码；原型条件作为对比实验（见第 8 节）。

```python
MAX_RACK = 8
NUM_POS = MAX_RACK + 1                                   # 0 = 底板层，1..MAX_RACK = 导轨层

class RackHead(nn.Module):
    def __init__(self, in_dim=2048, proto_dim=256, cond_dim=64, use_proto=False):
        super().__init__()
        self.use_proto = use_proto
        self.cond = nn.Linear(proto_dim, cond_dim) if use_proto else None
        self.mlp = nn.Sequential(nn.Linear(in_dim + (cond_dim if use_proto else 0), 512), nn.GELU())
        self.level = nn.Linear(512, NUM_POS)

    def forward(self, feat, valid, proto=None):          # valid: [B, NUM_POS] bool
        x = torch.cat([feat, self.cond(proto)], -1) if self.use_proto else feat
        logits = self.level(self.mlp(x))
        return logits.masked_fill(~valid, float("-inf"))    # 推理：level = argmax


def build_valid_mask(rack_count, floor_usable):          # rack_count: [B] long，floor_usable: [B] bool
    k = torch.arange(NUM_POS, device=rack_count.device)[None, :]
    valid = k <= rack_count[:, None]                     # 屏蔽超过该型号层数的导轨层
    valid[:, 0] &= floor_usable                          # 不能直接放底板的机型屏蔽第 0 类
    return valid                                         # 非一体机样本填 rack_count=MAX_RACK、floor_usable=True
```

**训练细节**

- 参与训练的样本：`is_oven = 1` 且层位不为 null。非一体机样本即使有附件也不参与；推理时由 Is-Oven 门控。
- **训练时的掩码一律用真实型号的 `rack_count`。** 在条件实验里，把原型替换成检索结果只改条件向量，不改掩码。否则一旦检索到层数更少的型号，真实层号会被屏蔽掉，loss 直接变成 inf。这个特性也有好处：标注层号超出 rack_count 时 loss 为 inf，可以借此发现标注错误。
- 基线 loss 用普通 CE。
- 按型号均衡采样，避免层数少的机型被数据量淹没。
- 先观察逐层准确率，再决定要不要加类别权重（例如频率倒数的平方根）。
- 可选：邻层软标签，只在导轨层之间做平滑。第 0 类不参与，因为它包含"不用附件"的情形，与第 1 层的区别不只是高度。如果评估时发现混淆集中在"附件平放底板"和第 1 层之间，可以再重新考虑。

```python
def rack_loss(logits, target, rack_count, smooth=0.0):
    # logits: [B, NUM_POS]，已掩码；target: [B]，取值 0..rack_count
    if smooth == 0:
        return F.cross_entropy(logits, target)
    soft = F.one_hot(target, logits.size(1)).float() * (1 - smooth)
    for d in (-1, 1):
        nb = target + d
        ok = (target >= 1) & (nb >= 1) & (nb <= rack_count)   # 只在导轨层之间平滑
        idx = ok.nonzero(as_tuple=True)[0]
        soft[idx, nb[idx]] += smooth / 2
    soft = soft / soft.sum(-1, keepdim=True)
    logp = F.log_softmax(logits, -1).masked_fill(torch.isinf(logits), 0)
    return -(soft * logp).sum(-1).mean()
```

**数据增强约束**（适用于一体机样本）

- 不做上下翻转。
- 不做大角度旋转或大幅的竖直平移。
- 随机裁剪不能切掉底板和导轨。
- 左右翻转、颜色和亮度扰动都可以。

### 4.6 Accessory Head

- 9 个类别（已去掉"无"）：玻璃蒸烤盘、脆烤盘、微波专用烤架、烤架、炸烤网架、烤盘、有孔蒸盘、无孔蒸盘、小炸篮。
- 多标签，用 BCE，**全部样本参与**（一体机内外都会出现附件）。
- 稀有类用 `pos_weight` 或 focal loss，每个类在验证集上单独调阈值。
- 只有在"一体机内部且型号已知"时，才按 Device Profile 屏蔽不支持的附件。

### 4.7 Container Head

- 10 个类别（已去掉"无"）：玻璃容器、塑料容器、陶瓷容器、金属容器、泡沫容器、木竹容器、纸质容器、油纸、保鲜膜、铝箔。
- 多标签，用 BCE，全部样本参与。
- 油纸、保鲜膜、铝箔属于覆盖物，可以和容器同时标注。
- 安全相关类别（金属容器、铝箔、泡沫容器）按召回率目标来定阈值，不用统一的 0.5。

### 4.8 Food Head

- 二分类，用 BCE，全部样本参与。

---

## 5. Loss

| 分支 | Loss | 参与样本 | 初始权重 |
|---|---|---|---|
| Is-Oven | BCE | 全部 | 0.5 |
| Proj（型号） | SupCon / ArcFace | is_oven = 1 | 1.0 |
| Food | BCE | 全部 | 0.5 |
| Container | BCE（pos_weight） | 全部 | 1.0 |
| Accessory | BCE（pos_weight） | 全部 | 1.0 |
| 层位 | CE（层数掩码） | is_oven = 1 且层位不为 null | 1.0 |

**规则**

- 带掩码的 loss 按有效样本数取平均。
- batch 内没有有效样本时返回 0；使用 DDP 时要开启 `find_unused_parameters=True`。
- 分别记录每一项 loss 和各个头的梯度范数，据此调整权重。
- 标签里的 null 在 dataloader 中填一个占位值，最终由掩码排除。

```python
def masked_loss(fn, pred, target, valid):
    if valid.any():
        return fn(pred[valid], target[valid])
    return torch.zeros((), device=pred.device)

is_oven   = batch["is_oven"]                           # [B] bool
has_level = is_oven & batch["rack_level_known"]        # 一体机内部且非空腔

loss = (0.5 * bce(oven_logit, is_oven.float())
      + 1.0 * masked_loss(metric_loss, proj_feat, device_id, is_oven)
      + 0.5 * bce(food_logit, food.float())
      + 1.0 * container_bce(container_logits, container)
      + 1.0 * accessory_bce(accessory_logits, accessory)
      + 1.0 * masked_loss(ce, rack_logits, rack_level, has_level))
```

---

## 6. 数据格式

**一体机内部 · 导轨层**

```json
{
  "image": "a.jpg",
  "is_oven": true,
  "device_model": "C9277A",
  "food_exist": true,
  "container_type": ["塑料容器"],
  "accessory_type": ["小炸篮"],
  "rack_level": 3
}
```

**一体机内部 · 底板层**

```json
{
  "image": "b.jpg",
  "is_oven": true,
  "device_model": "DB677",
  "food_exist": true,
  "container_type": ["陶瓷容器"],
  "accessory_type": [],
  "rack_level": 0
}
```

**非一体机**

```json
{
  "image": "c.jpg",
  "is_oven": false,
  "device_model": null,
  "food_exist": true,
  "container_type": ["陶瓷容器"],
  "accessory_type": ["烤盘"],
  "rack_level": null
}
```

**约定**

- 空列表 `[]` 表示"无"；`null` 表示该项不适用。
- `rack_count`、`rack_ratio`、`rack_exist` 不写进标注：前两者从配置表推导，后者由"层位 ≥ 1"推出。
- 多标签向量中不包含"无"这一类。例如"小炸篮 + 烤盘"编码为 `[0,0,0,0,0,1,0,0,1]`。

**入库校验**

- 所有类别名都必须在类别表中。
- `is_oven = true` 时，`device_model` 必填。
- `rack_level` 必须在 0～rack_count 范围内；如果该型号 `floor_usable = false`，则不能为 0。
- `rack_level ≥ 1` 时，`accessory_type` 不能为空。
- `is_oven = true` 且腔内有任何物品（食物、容器或附件）时，`rack_level` 必填；空腔时为 null。
- `is_oven = false` 时，`device_model` 和 `rack_level` 都必须为 null。

---

## 7. 标注规范要点

- **层位**
  - 0 表示底板层：物品直接放在底板上，不管有没有用附件。
  - 1～N 表示导轨层，从下往上数。
  - 有物品的一体机内部图片必须标出确定的层位；空腔图片的层位标 null。
- **主要附件**：默认指放有食物的那个附件。多个附件都放了食物时，判定规则待补充。
- **非一体机图片**：同样要标附件，没有附件就标空列表。
- **容器与覆盖物**：油纸、保鲜膜、铝箔是覆盖物，可以和容器同时标，例如"玻璃容器 + 保鲜膜"。
- **铝箔**：需要明确只指铝箔纸，还是也包括铝箔盒。建议铝箔盒归入"金属容器"。
- **玻璃蒸烤盘与玻璃容器**：随机附带的算附件，用户自己的器皿算容器，标注规范里附上对照图。
- **数据划分**：按拍摄批次或整机划分训练、验证、测试集，不要按单张图片随机划分，以防近似重复的图片造成泄漏。

---

## 8. 训练流程

1. **阶段 1（基线）**：冻结骨干，所有头和 Proj Head 联合训练。Rack Head 不加型号条件，只用层数掩码。
2. **阶段 2**：用 Proj Head 建特征库，标定 `tau`；在验证集上为多标签的每一类调阈值，同时确定层位的置信度阈值。
3. **阶段 3（对比实验）**：冻结其余部分，训练带原型条件的 Rack Head。训练时以 20% 的概率把原型换成检索结果，掩码始终用真实的 rack_count。与基线对比，提升不明显就保留基线。
4. **阶段 4（可选）**：解冻骨干的最后几层，或用 LoRA 微调。之后要重建特征库，并重新标定 `tau` 和各项阈值。

---

## 9. 评估

| 任务 | 指标 |
|---|---|
| Is-Oven | 准确率、召回率 |
| 型号 | top-1 准确率、混淆矩阵 |
| 型号（新型号能力） | 留一型号验证：库外型号的拒识率；把它加入特征库后的识别率 |
| 层位 | 按型号统计 (N+1)×(N+1) 混淆矩阵；精确层准确率；导轨层 ±1 准确率；底板层 vs 导轨层的二分类准确率 |
| 附件 | 每类的 P / R / AP 和 mAP，一体机内外分开统计 |
| 容器 | 每类的 P / R / AP 和 mAP；安全相关类别单独统计召回率 |
| 食物 | 准确率、F1 |
| 端到端 | 型号识别错误时层位的出错比例（衡量级联误差） |

**留一型号验证**：训练 Proj Head 时去掉一个型号，只把它的参考图放进特征库，测它的识别率；同时测它在加入特征库之前能否被判为"未知型号"。每个型号轮流做一遍。

---

## 10. 新增型号流程

1. 拍 50～200 张参考图，覆盖空腔、放附件、放食物、灯亮/灯暗、底板层和各导轨层。
2. 提取特征后调用 `gallery.add`；在 Device Profile 中补充 `rack_count`、`floor_usable`、`accessories` 和 `cavity_group`。
3. 用几十张留出图检查 top-1 准确率，重点看它和外观相近型号之间的混淆，必要时调整 `tau`。
4. 收集一小批标了层位的图片，只微调 Rack Head。骨干、Proj Head 和特征库都不用动。

---

## 11. 待确认事项

- [ ] 推理时设备型号是否已知（例如摄像头内置在一体机里）。如果已知，直接用真实型号，跳过检索。
- [ ] `MAX_RACK` 的最终取值。
- [ ] 多个附件都放了食物时，如何确定主要附件。
- [ ] 各型号支持的附件列表、是否有内腔完全相同的型号、是否有机型不能直接放底板（`floor_usable`）。
- [ ] 安全相关类别（金属、铝箔、泡沫）的召回率目标。
