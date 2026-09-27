# utils/logging.py
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional


_SHANGHAI_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


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
    统一日志初始化函数
    """
    # ===== 1. 先保证 root logger 有 handler（兜底，debug 必须）=====
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        formatter_cls = _ShanghaiFormatter if use_shanghai_time else logging.Formatter
        root_fmt = formatter_cls(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
        rh = logging.StreamHandler()
        rh.setLevel(level)
        rh.setFormatter(root_fmt)
        root.addHandler(rh)

    # ===== 2. 再配置你自己的 app logger =====
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 命名 logger 已有自己的 handler，关闭向 root 传播，避免同一条日志重复输出。
    logger.propagate = False

    formatter_cls = _ShanghaiFormatter if use_shanghai_time else logging.Formatter
    fmt = formatter_cls(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    # 清理旧 handler
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
