# dinov3_machine

基于 DINOv3 的一体机多任务视觉识别模型。输入一张图片，输出：是否一体机内部、设备型号（检索式，库外型号输出"未知型号"）、是否有食物、容器（多标签 10 类）、附件（多标签 13 类）和单个层位（0 = 底板层，1～N = 导轨层）。

整体方案见 [docs/oven-multitask-design.md](docs/oven-multitask-design.md)，使用方法见 [script/0-使用方法.md](script/0-使用方法.md)。

## 快速开始

正式训练使用 Linux + NVIDIA GPU，建议 Python 3.11。先创建并激活环境，安装与训练机驱动匹配的 CUDA 版 PyTorch 和 torchvision，然后在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
git submodule update --init third_party/dinov3

# 确认预训练权重已复制到该位置
test -f data/model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

# 原始标注及图片放在 data/00-origin_data，生成到 data/01-train_data/<日期>
python script/01-genTrainData.py
# 将 configs/default_oven.yaml 的 data.root 改为上一步实际打印的目录
python script/02-checkTrainData.py --config configs/default_oven.yaml
python train.py --config configs/default_oven.yaml --device cuda
```

也可用 `--src <原始数据目录> --dst <训练数据根目录>` 指定输入输出。生成器按型号分层划分 train / val / test，把同图多份标注及文件内容相同的图片绑定到同一划分；同日重跑会另建目录，保留已有结果。本次已生成 `data/01-train_data/20261017`，训练 / 验证 / 测试分别为 7275 / 910 / 908 张，默认配置已指向该目录，硬错误默认报错。

`02-checkTrainData.py` 使用真实训练解析器和 Dataset 入库校验，读取首个训练 batch，并检查跨划分的相同图片内容；报告默认写到 `output/preflight/data_check.json`。数据预检通过后再启动 GPU 训练。训练结束自动用最佳权重建库、标定阈值并评估测试集，输出在 `output/oven/<YYMMDD_HHMMSS>/`。同秒启动另建后缀目录；指定 `--output_dir` 时要求目录不存在或为空，续训用 `--resume <run>/ckpt_last.pt`。

需要逐型号重新训练验证时，可在阶段 2 完成后运行 `python script/4-留一型号验证.py --run output/oven/<run> --device cuda`。这是可选步骤，每个内腔组都要重训，耗时约为多次训练。

`device_model="未知"` / `"未知型号"` 表示明确非本司的设备内腔：保留 `is_oven=true`，参与可靠的食物、容器和附件训练，不参与公司型号和层位训练、不进入型号特征库。PK 采样使用独立的 `sampler.unknown_oven_ratio` 配额；`non_oven_ratio` 仍只表示非内腔图片。图片标注中显式出现的 `null` 表示确认没有（食物为 false，容器/附件为空列表）；缺失字段和“遮挡”不监督。层位的 `null` / `["无"]` 不训练，数字 `0` 才是底板层。设备配置表中的 `accessories:null` 仍表示支持列表尚未配置。

当前 Rack Head 只预测单个层位。原始数据中的 482 张多层图片保留食物、容器、附件等标签，生成后层位设为 `null`，原层位保存在 `training_preparation.multi_rack_level`，不参与层位 loss。`device_profile.json` 的部分底板可用性、内腔分组、附件支持仍是占位值，正式训练前应核实。

本批 `CQ09-i9` 共 13 张、外部未知共 11 张，两者在验证集和测试集均各只有 1 张，相关指标及拒识阈值不足以证明泛化效果。图片分层随机划分仍可能将同一视频的相邻帧分到不同集合；应另做按拍摄场次或视频分组的独立测试。

加载预训练骨干时，所有参数和持久缓冲区必须完整且形状匹配；缺失或不匹配会直接报错，停止训练。支持自动移除常见 checkpoint 前缀；额外的训练头权重不加载。

训练完成后的推理与导出均使用固定 `batch_size=1`，输入目录中的图片逐张处理：

```bash
# 先修改 5-infer-dir.py 顶部 RUN_DIR、INPUT_PATH、DEVICE，再运行（无命令行参数）
python script/5-infer-dir.py
python script/6-export_onnx.py --run output/oven/<run> --check_input <图片或目录>
python script/7-infer_onnx.py --onnx_dir output/oven/<run>/onnx --input <图片或目录>
```

`5-infer-dir.py` 是带标注对照的目录评估工具，默认把可视化写到 `<run>/vis/<输入目录名>/`；无同名 JSON 的图片不保存对照图。ONNX 推理支持图片或目录输入，结果保存为 JSON。

ONNX 输入固定为 `[1, 3, H, W]`，导出时使用传统 PyTorch ONNX 导出器，无需 `onnxscript`。`--check_images` 控制检查图片数量，每张均以 batch=1 验证；合并或输出对齐检查失败时会报错停止。训练和验证的 batch size 仍由训练配置决定。

## 目录结构
```text
├── configs
│   ├── default_oven.yaml        # 训练配置（模型、类别表、数据、增强、采样、loss、训练、评估）
│   └── device_profile.json      # 设备配置表：rack_count / floor_usable / cavity_group / accessories
├── dino_finetune
│   ├── config.py                # 配置读取、校验与默认值
│   ├── labels.py                # 标注解析（1.x 导出格式 / 规范格式）与入库校验
│   ├── device.py                # Device Profile、检索特征库 DeviceGallery、tau 标定
│   ├── data.py                  # 预处理与增强、Dataset、PK 采样
│   ├── losses.py                # 掩码 loss、SupCon / ArcFace、层位 CE（邻层软标签）、多任务加权
│   ├── metrics.py               # 各头指标、型号检索、级联误差、阶段 2 阈值标定
│   ├── model
│   │   ├── oven.py              # AttnPool、各任务头、层数掩码、OvenMultiTaskModel
│   │   └── lora.py              # Q/V LoRA 注入与权重合并（默认开启）
│   └── utils
│       ├── ckpt.py              # 加载 DINOv3 骨干与预训练权重
│       └── training_monitor.py  # 指标 JSON 与训练曲线
├── script                       # 启动脚本与工具
└── train.py                     # 训练入口：阶段 1 联合训练 + 阶段 2 建库、标定与测试
```

## 与设计文档的对应

- 已实现：各任务头联合训练、可选 Q/V LoRA（默认开启）、建特征库、标定 tau / 逐类阈值 / 层位置信度阈值，以及 PT 推理、固定 batch=1 的 ONNX 导出和推理。`model.use_lora: false` 可训练冻结骨干基线，再用 `--init` 加载基线并开启 LoRA；`model.lora_last_n_blocks` 控制注入范围。
- tau 优先用验证集中的库内公司设备和真实外部设备标定；没有外部样本时，用“把该型号所在 cavity_group 临时移出特征库”近似留一型号验证，Proj Head 不重训。两种标定均不可用时，使用配置默认值。
- `script/4-留一型号验证.py` 实现了逐 cavity_group 留出重训、拒识与入库后识别评估。
- 尚未实现：阶段 3（原型条件 Rack Head 对比实验）、直接解冻骨干原始权重；不支持 `model.unfreeze_last_n_blocks`。
