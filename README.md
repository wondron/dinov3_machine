# dinov3_machine

基于 DINOv3 的一体机多任务视觉识别模型。输入一张图片，输出：是否一体机内部、设备型号（检索式，库外型号输出"未知型号"）、是否有食物、容器（多标签 10 类）、附件（多标签 9 类）和层位（0 = 底板层，1～N = 导轨层）。

整体方案见 [docs/oven-multitask-design.md](docs/oven-multitask-design.md)，使用方法见 [script/0-使用方法.md](script/0-使用方法.md)。

## 快速开始
```bash
pip install -r requirements.txt
# 需要：third_party/dinov3（DINOv3 源码）和 data/model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth（预训练权重）
python train.py --config configs/default_oven.yaml --device cuda
```

加载预训练骨干时，所有参数和持久缓冲区必须完整且形状匹配；缺失或不匹配会直接报错，停止训练。支持自动移除常见 checkpoint 前缀；额外的训练头权重不加载。

训练完成后的推理与导出均使用固定 `batch_size=1`，输入目录中的图片逐张处理：

```bash
python script/5-infer.py --run output/oven/<run> --input <图片或目录>
python script/6-export_onnx.py --run output/oven/<run> --check_input <图片或目录>
python script/7-infer_onnx.py --onnx_dir output/oven/<run>/onnx --input <图片或目录>
```

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
- tau 的标定用"把该型号所在 cavity_group 临时移出特征库"近似留一型号验证，Proj Head 不重训；特征库里少于 2 个 cavity_group 时无法标定，使用配置默认值。
- `script/4-留一型号验证.py` 实现了逐 cavity_group 留出重训、拒识与入库后识别评估。
- 尚未实现：阶段 3（原型条件 Rack Head 对比实验）、直接解冻骨干原始权重；不支持 `model.unfreeze_last_n_blocks`。
