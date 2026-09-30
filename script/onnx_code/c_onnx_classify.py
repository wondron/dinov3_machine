"""可复制到其他项目的一体机 ONNX 推理入口，仅依赖 NumPy、OpenCV、ONNX Runtime。"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import onnxruntime as ort

if __package__:
    from .onnx_postprocess import OvenPostprocessor, load_device_profile, load_gallery_bundle
else:
    from onnx_postprocess import OvenPostprocessor, load_device_profile, load_gallery_bundle

logger = logging.getLogger(__name__)

OUTPUT_NAMES = (
    "is_oven_prob", "food_prob", "container_prob", "accessory_prob", "rack_raw", "proj", "cls",
)
INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}
ImageInput = bytes | bytearray | memoryview | str | os.PathLike | np.ndarray


class OnnxClassifier:
    """加载一次部署包，重复调用 detect；返回多任务结果字典。

    config_data 示例：
        {"CLASS_CONFIG": {"onnx_dir": "/models/oven/onnx", "provider": "auto"}}
    也支持 CLASS_CONFIG.model_path 指向部署包中的 oven.onnx。
    图片支持编码字节、文件路径、OpenCV BGR uint8 数组。每次推理固定 batch=1。
    """

    def __init__(self, config_data: Mapping[str, Any]) -> None:
        self.isInit = False
        if not isinstance(config_data, Mapping):
            raise TypeError("config_data 必须是配置字典")
        cfg = config_data.get("CLASS_CONFIG", config_data)
        if not isinstance(cfg, Mapping):
            raise TypeError("CLASS_CONFIG 必须是配置字典")
        location = cfg.get("onnx_dir") or cfg.get("model_path")
        if not location:
            raise ValueError("请配置 CLASS_CONFIG.onnx_dir 或 CLASS_CONFIG.model_path")
        location = Path(location).expanduser().resolve()
        if cfg.get("onnx_dir") or location.is_dir():
            self.bundle_dir, model_path = location, location / "oven.onnx"
        else:
            self.bundle_dir, model_path = location.parent, location
        self.model_path = str(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX 模型不存在：{model_path}")

        self.meta = self._read_json("meta.json")
        self.calibration = self._read_json("calibration.json")
        galleries, gallery_meta = load_gallery_bundle(self.bundle_dir)
        versions = [obj.get("fingerprint") for obj in (self.meta, self.calibration, gallery_meta)]
        if any(not isinstance(v, str) or not v for v in versions) or len(set(versions)) != 1:
            raise RuntimeError(f"部署包内的模型 / 阈值 / 特征库版本不一致或缺少 fingerprint：{versions}")
        self.profile = load_device_profile(
            self.bundle_dir / "device_profile.json",
            max_rack=self.meta["max_rack"],
            accessory_classes=self.meta["accessory_classes"],
        )
        self.post = OvenPostprocessor(
            calibration=self.calibration,
            profile=self.profile,
            container_classes=self.meta["container_classes"],
            accessory_classes=self.meta["accessory_classes"],
            galleries=galleries,
        )
        self.device_model = cfg.get("device_model")
        self.with_scores = bool(cfg.get("with_scores", False))
        self._validate_device_model(self.device_model)
        self._load_preprocess()
        for key in ("input_size", "min_conf"):
            if key in cfg:
                logger.warning("忽略旧分类配置 %s；尺寸和各任务阈值由部署包提供", key)

        self.provider = str(cfg.get("provider", "auto")).lower()
        if self.provider not in ("auto", "cpu", "cuda"):
            raise ValueError("CLASS_CONFIG.provider 必须是 auto、cpu 或 cuda")
        self.device_id = int(cfg.get("device_id", 0))
        if self.device_id < 0:
            raise ValueError("CLASS_CONFIG.device_id 不能小于 0")
        self._create_session()
        self._validate_session()
        self.isInit = True
        logger.info("已加载 ONNX 部署包：%s，版本=%s，providers=%s",
                    self.bundle_dir, versions[0], self.session.get_providers())

    def _read_json(self, name: str) -> dict[str, Any]:
        path = self.bundle_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"部署包缺少文件：{path}，请复制完整导出目录")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{path} 必须是 JSON 对象")
        return value

    def _load_preprocess(self) -> None:
        inp = self.meta["input"]
        if (inp.get("batch_size", 1), inp.get("color", "RGB"), inp.get("layout", "NCHW")) != (1, "RGB", "NCHW"):
            raise ValueError("部署包必须使用 batch=1、RGB、NCHW 输入")
        dims = inp["img_dim"]
        if len(dims) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dims):
            raise ValueError("meta.input.img_dim 必须为正整数 [高, 宽]")
        self.input_size = tuple(dims)
        self.input_name = inp["name"]
        self.mean = np.asarray(inp["mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(inp["std"], dtype=np.float32).reshape(1, 1, 3)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("meta.input.mean/std 必须有限，std 必须大于 0")
        interp = str(inp["img_interp"]).lower().strip()
        if interp not in INTERPOLATIONS:
            raise ValueError(f"不支持的 meta.input.img_interp：{interp}")
        self.interp = INTERPOLATIONS[interp]
        self.output_names = list(self.meta["outputs"])
        if len(self.output_names) != len(OUTPUT_NAMES) or set(self.output_names) != set(OUTPUT_NAMES):
            raise ValueError(f"meta.outputs 必须包含这七个输出且不重复：{OUTPUT_NAMES}")

    def _create_session(self) -> None:
        available = ort.get_available_providers()
        wanted = {"auto": ("CUDAExecutionProvider", "CPUExecutionProvider"),
                  "cuda": ("CUDAExecutionProvider",), "cpu": ("CPUExecutionProvider",)}[self.provider]
        providers = [name for name in wanted if name in available]
        if not providers:
            raise RuntimeError(f"ONNX Runtime 没有可用的 {wanted}，当前可用：{available}")
        options = [{"device_id": self.device_id} if p == "CUDAExecutionProvider" else {} for p in providers]
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(
                self.model_path, sess_options=session_options, providers=providers, provider_options=options,
            )
        except Exception:
            if self.provider != "auto" or "CUDAExecutionProvider" not in providers or "CPUExecutionProvider" not in available:
                raise
            logger.warning("CUDA 会话初始化失败，尝试 CPU 推理", exc_info=True)
            self.session = ort.InferenceSession(
                self.model_path, sess_options=session_options, providers=["CPUExecutionProvider"],
            )
        self.providers = self.session.get_providers()
        if self.provider == "cuda" and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError("指定了 cuda，但 CUDA 会话初始化失败；请检查 ONNX Runtime / CUDA 环境")
        if self.provider == "auto" and "CUDAExecutionProvider" not in self.providers:
            logger.info("使用 CPU 推理")

    def _validate_session(self) -> None:
        inputs = self.session.get_inputs()
        expected_shape = [1, 3, *self.input_size]
        if len(inputs) != 1 or inputs[0].name != self.input_name or inputs[0].type != "tensor(float)" or inputs[0].shape != expected_shape:
            raise ValueError(f"ONNX 输入与 meta.json 不一致：应为 {self.input_name} float32 {expected_shape}")
        outputs = {node.name: node for node in self.session.get_outputs()}
        if set(outputs) != set(self.output_names):
            raise ValueError("ONNX 输出名称与 meta.outputs 不一致")
        expected = {
            "is_oven_prob": [1], "food_prob": [1],
            "container_prob": [1, len(self.meta["container_classes"])],
            "accessory_prob": [1, len(self.meta["accessory_classes"])],
            "rack_raw": [1, self.meta["max_rack"] + 1],
        }
        for name, shape in expected.items():
            if outputs[name].shape != shape or outputs[name].type != "tensor(float)":
                raise ValueError(f"ONNX 输出 {name} 应为 float32 {shape}，实际为 {outputs[name].shape}")
        for name in ("proj", "cls"):
            node = outputs[name]
            if node.type != "tensor(float)" or len(node.shape) != 2 or node.shape[0] != 1:
                raise ValueError(f"ONNX 特征输出 {name} 必须为 float32 [1, D]")

    def _validate_device_model(self, device_model: str | None) -> None:
        if device_model is not None and device_model not in self.profile:
            raise ValueError(f"型号 {device_model} 不在 Device Profile 中")

    @staticmethod
    def _read_bgr(image_data: ImageInput) -> np.ndarray:
        if isinstance(image_data, np.ndarray):
            image = image_data
        else:
            if isinstance(image_data, (str, os.PathLike)):
                image_data = Path(image_data).read_bytes()
            if not isinstance(image_data, (bytes, bytearray, memoryview)):
                raise TypeError("图片必须是编码字节、文件路径或 BGR uint8 NumPy 数组")
            encoded = np.frombuffer(image_data, dtype=np.uint8)
            if not encoded.size:
                raise ValueError("图片字节为空")
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法解码图片，请传入 JPEG/PNG 等图片编码字节")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("NumPy 图片必须为非空 BGR uint8 [高, 宽, 3]，像素范围 0～255")
        return image

    def _preprocess(self, image_data: ImageInput) -> np.ndarray:
        image = cv2.cvtColor(self._read_bgr(image_data), cv2.COLOR_BGR2RGB)
        height, width = self.input_size
        image = cv2.resize(image, (width, height), interpolation=self.interp)
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)

    def warmup(self, times: int = 1) -> None:
        """实际执行 times 次模型前向；建议在服务启动时调用。"""
        if not self.isInit:
            raise RuntimeError("模型未初始化")
        if isinstance(times, bool) or not isinstance(times, int) or times < 0:
            raise ValueError("times 必须为非负整数")
        image = np.zeros((1, 3, *self.input_size), dtype=np.float32)
        for _ in range(times):
            self.session.run(self.output_names, {self.input_name: image})

    def detect(
        self,
        image_data: ImageInput | None,
        device_model: str | None = None,
        with_scores: bool | None = None,
    ) -> dict[str, Any]:
        """识别单张图片。detect(bytes, None) 可沿用附件调用形式。

        None 图片返回 {}；其余无效输入抛异常。
        device_model=None 使用配置中的默认型号；配置也为 None 时检索型号。
        返回普通 Python 类型，可直接 json.dumps；本方法不修改实例配置。
        """
        if not self.isInit:
            raise RuntimeError("模型未初始化")
        if image_data is None:
            return {}
        model = self.device_model if device_model is None else device_model
        self._validate_device_model(model)
        image = self._preprocess(image_data)
        values = self.session.run(self.output_names, {self.input_name: image})
        outputs = dict(zip(self.output_names, values))
        return self.post(
            outputs, device_model=model,
            with_scores=self.with_scores if with_scores is None else with_scores,
        )[0]
