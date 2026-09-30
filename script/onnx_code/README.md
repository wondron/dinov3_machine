# 其他项目接入 ONNX 推理

这是 `script/7-infer_onnx.py` 的独立部署版本。把整个 `onnx_code` 文件夹复制到其他项目即可使用，不需要 `dino_finetune`、PyTorch、训练权重或原项目的 `app.domain.common.image_utils`。

保留附件的 `OnnxClassifier(config)`、`isInit`、`warmup(times)`、`detect(image_data, None)` 调用形式，返回本项目的多任务识别结果。

## 1. 准备代码和模型

代码文件：

```text
onnx_code/
  __init__.py
  c_onnx_classify.py    # 配置、会话、预处理和 detect 接口
  onnx_postprocess.py  # NumPy 特征检索、设备配置校验、任务后处理
  example.py           # 单图 / 目录命令行示例
  requirements.txt
  README.md
```

先在训练项目通过 `script/6-export_onnx.py` 导出部署包，再将**完整部署目录**复制到业务项目，例如：

```text
your_project/
  onnx_code/
  models/oven/
    oven.onnx
    meta.json
    calibration.json
    device_profile.json
    gallery.json
    gallery_proj.npy       # 具体文件以 gallery.json 中的 features 为准
    gallery_cls.npy
    export_check.json
```

如果导出目录还有 ONNX 外部权重文件，也要一起复制。推理读取 `gallery.json` 引用的特征文件；`export_check.json` 是导出检查记录。代码会检查模型元数据、校准参数、特征库元数据的 `fingerprint` 一致性，以及 ONNX 输入输出与 `meta.json` 是否匹配。

运行环境为 Python 3.10 或更高版本。CPU 安装：

```bash
python -m pip install -r onnx_code/requirements.txt
```

GPU 环境使用 `onnxruntime-gpu` 替代 requirements 中的 `onnxruntime`，并配置对应的 CUDA / cuDNN 环境。同一环境只保留一个 ONNX Runtime 包。已安装 `opencv-python` 的业务项目可继续使用，不必再装提供相同 `cv2` 模块的 headless 包。部署运行不需要安装 `onnx`。

## 2. 按附件形式接入

```python
from pathlib import Path
from onnx_code import OnnxClassifier

PROJECT_DIR = Path(__file__).resolve().parent
config = {
    "CLASS_CONFIG": {
        "model_path": str(PROJECT_DIR / "models/oven/oven.onnx"),
        "provider": "auto",
        "device_id": 0,
        "device_model": None,
        "with_scores": False,
    }
}

# 服务启动时初始化一次，后续请求复用。
clf = OnnxClassifier(config)
clf.warmup(times=1)

image_bytes = (PROJECT_DIR / "test.jpg").read_bytes()
result = clf.detect(image_bytes, None)
print(result)

# 型号已知时，跳过特征库检索；名称必须存在于 device_profile.json。
# result = clf.detect(image_bytes, device_model="C87-i7Pro", with_scores=True)
```

也可以使用 `"onnx_dir": str(PROJECT_DIR / "models/oven")` 替代 `model_path`。两者同时配置时优先使用 `onnx_dir`。相对路径以业务程序的**当前工作目录**为基准，建议如上构造绝对路径。

如果把代码放在 `app/domain/steps/onnx_code/`，则使用：

```python
from app.domain.steps.onnx_code import OnnxClassifier
```

如果沿用附件的 `app.domain.steps.c_onnx_classify` 导入路径，将 `c_onnx_classify.py` 和 `onnx_postprocess.py` 一起复制到 `app/domain/steps/`，即可：

```python
from app.domain.steps.c_onnx_classify import OnnxClassifier
```

不要只复制入口文件，它需要同目录中的 `onnx_postprocess.py`。

## 3. 配置和图片输入

| 配置项 | 默认值 | 含义 |
| --- | --- | --- |
| `onnx_dir` / `model_path` | 必填其一 | 完整部署目录 / 目录中的 ONNX 文件 |
| `provider` | `auto` | `auto` 优先 CUDA，无法初始化时回退 CPU；`cpu` 使用 CPU；`cuda` 不可用时抛错 |
| `device_id` | `0` | CUDA 设备编号 |
| `device_model` | `None` | 默认已知型号；`None` 表示检索特征库 |
| `with_scores` | `False` | 是否返回各任务的原始概率 |

`detect` 的 `device_model` 非空时覆盖默认型号；传 `None` 时使用配置中的默认型号。需要按图片自动检索时，配置中的 `device_model` 也应为 `None`。`with_scores=None` 使用配置值，传 `True` / `False` 可逐次覆盖。

```python
import cv2

# 图片文件的编码字节（如 HTTP 上传后得到的 bytes），也支持 bytearray / memoryview。
result = clf.detect(image_bytes)

# 文件路径，支持 Windows 中文路径。
result = clf.detect(PROJECT_DIR / "图片.png")

# OpenCV 数组：BGR、uint8、[H, W, 3]、像素范围 0～255。
image_bgr = cv2.imread(str(PROJECT_DIR / "test.jpg"))
result = clf.detect(image_bgr)
```

数组不要预先转 RGB、缩放、归一化或添加 batch 维度。`detect(None)` 沿用附件习惯返回 `{}`；无效编码字节、缺失文件、错误数组格式或未知的指定型号会抛异常。`warmup` 会实际执行模型前向，`times=0` 可跳过。

输入固定为 batch=1；遍历图片逐次调用 `detect` 即可。推理方法不修改实例配置，初始化后可以复用同一个实例；业务调用期间不要修改其公开配置对象。

## 4. 返回结果

以下是字段示意，实际型号、类别和分数由部署包及图片决定：

```json
{
  "is_oven": true,
  "device_model": "C87-i7Pro",
  "device_score": 0.96,
  "food_exist": true,
  "container_type": ["bowl"],
  "accessory_type": ["tray"],
  "rack_level": 2,
  "rack_level_score": 0.91,
  "low_confidence": false
}
```

- 非一体机：`device_model="无"`，`rack_level=null`；仍输出食物、容器、附件。
- 未知型号：`device_model="未知型号"`，`pending_gallery=true`，`rack_level=null`。业务方可据此收集待补库图片；`detect` 本身不复制文件。
- 已知型号：屏蔽不支持的附件，在该型号有效层位上重新 softmax 后选择层位。空腔层位为 `null`；低层位置信度，或层位 ≥ 1 却没有附件时，`low_confidence=true`。
- `device_score`：检索成功时为获胜内腔组的加权票数占比；未知型号时为 top-1 余弦相似度（无可用特征库时为 `null`）；指定型号时为 `1.0`。
- `with_scores=True` 会增加 `scores`，包含 `is_oven`、`food`、按类别名索引的 `container` 和 `accessory` 概率。所有返回值都可直接 `json.dumps`。

本模型不是附件中的单头 TopK 分类模型，因此返回值不是 `{类别索引: 分数}`。类别、每项阈值、尺寸、均值和标准差全部来自部署包；旧配置 `min_conf` / `input_size` 会被忽略并记日志。不读取 `.plk`，也不提供原附件判断 `shu` 类的 `isShu()`，下游应按上述多任务字段读取结果。

预处理与原脚本一致：BGR → RGB，按 `[高, 宽]` 直接 resize，除以 255，再按 `meta.json` 的 mean/std 归一化，输出 float32 NCHW。NumPy 检索复现原有的余弦相似度和按内腔组加权投票；多个样本相似度完全相同时，NumPy 与 PyTorch 的并列排序可能不同，阈值边界处也可能有浮点舍入差异。

## 5. 命令行验证

在本仓库运行：

```bash
python script/onnx_code/example.py --onnx_dir output/oven/<run>/onnx --input test.jpg --provider cpu --scores
python script/onnx_code/example.py --onnx_dir output/oven/<run>/onnx --input images/ --out predictions.json
```

复制到业务项目后运行：

```bash
python onnx_code/example.py --onnx_dir models/oven --input test.jpg --provider cpu
```

目录会递归扫描 `.jpg/.jpeg/.png/.bmp/.webp`，逐张推理。命令行输出 JSON 列表，额外附上每张图片的 `image` 路径；Python `detect` 接口返回单张结果字典。
