# backend/logger.py
"""
集中式日志管理模块
替代项目中散落的 print() 调试语句，使用 Python logging 标准库
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "transcriber", level: str = "INFO") -> logging.Logger:
    """
    创建并配置 logger
    
    Args:
        name: logger 名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # 格式：简洁但有用
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def setup_file_logger(name: str, log_file: Path, level: str = "DEBUG") -> logging.Logger:
    """
    创建带文件输出的 logger（用于打包环境）
    
    Args:
        name: logger 名称
        log_file: 日志文件路径
        level: 日志级别
    
    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def set_all_levels(level: str):
    """统一设置所有 logger 的日志级别"""
    lvl = getattr(logging, level.upper(), logging.INFO)
    for name in logging.Logger.manager.loggerDict:
        if name.startswith("transcriber"):
            logging.getLogger(name).setLevel(lvl)
