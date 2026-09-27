# dinov3_machine

基于 DINOv3 的一体机多任务视觉识别模型。输入一张图片，输出：是否一体机内部、设备型号（检索式，库外型号输出"未知型号"）、是否有食物、容器（多标签 10 类）、附件（多标签 9 类）和层位（0 = 底板层，1～N = 导轨层）。

整体方案见 [docs/oven-multitask-design.md](docs/oven-multitask-design.md)，使用方法见 [script/0-使用方法.md](script/0-使用方法.md)。

## 快速开始
```bash
pip install -r requirements.txt
# 需要：third_party/dinov3（DINOv3 源码）和 data/model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth（预训练权重）
python train.py --config configs/default_oven.yaml --device cuda
```

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
│   │   └── lora.py              # LoRA（阶段 4 可选）
│   └── utils
│       ├── ckpt.py              # 加载 DINOv3 骨干与预训练权重
│       └── training_monitor.py  # 指标 JSON 与训练曲线
├── script                       # 启动脚本与工具
└── train.py                     # 训练入口：阶段 1 联合训练 + 阶段 2 建库、标定与测试
```

## 与设计文档的对应
- 已实现：第 8 节阶段 1（冻结骨干联合训练）和阶段 2（建特征库、标定 tau / 逐类阈值 / 层位置信度阈值），阶段 4 通过配置 `model.unfreeze_last_n_blocks` / `model.use_lora` 加 `--init` 实现；第 9 节的评估指标写在 `val_report.json` / `test_report.json`。
- tau 的标定用"把该型号所在 cavity_group 临时移出特征库"近似留一型号验证，Proj Head 不重训；特征库里少于 2 个 cavity_group 时无法标定，使用配置默认值。
- 尚未实现：阶段 3（原型条件 Rack Head 对比实验）、完整的留一型号验证（每个型号轮流去掉后重训 Proj Head）、推理与导出脚本。
