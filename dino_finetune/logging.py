# dino_finetune/logging.py
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional


_SHANGHAI_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class _ShanghaiFormatter(logging.Formatter):
    """使用上海时区生成日志时间。"""

    def formatTime(self, record, datefmt=None):
        log_time = datetime.fromtimestamp(record.created, tz=_SHANGHAI_TIMEZONE)
        if datefmt:
            return log_time.strftime(datefmt)
        return log_time.isoformat(timespec="milliseconds")


def setup_logging(
    name: str = "app",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_shanghai_time: bool = False,
) -> logging.Logger:
    """
    统一日志初始化：handler 挂在 root logger 上，各模块的 logger 通过传播共用同一份输出（终端 + 可选日志文件）。
    重复调用会替换旧 handler，例如确定输出目录之后再补上日志文件。
    """
    formatter_cls = _ShanghaiFormatter if use_shanghai_time else logging.Formatter
    fmt = formatter_cls(fmt=_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
    return logger
