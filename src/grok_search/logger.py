"""Logging helpers shared by the MCP tools."""

import logging
from datetime import datetime
from typing import Any

from .config import config

logger = logging.getLogger("grok_search")
logger.setLevel(getattr(logging, config.log_level, logging.INFO))

_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

try:
    log_dir = config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"grok_search_{datetime.now().strftime('%Y%m%d')}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(getattr(logging, config.log_level, logging.INFO))
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)
except OSError:
    logger.addHandler(logging.NullHandler())


async def log_info(ctx: Any, message: str, is_debug: bool = False) -> None:
    """
    Send a log message to both the Python logger and the MCP context.

    Args:
        ctx: Optional MCP request context that can receive info messages.
        message: The message to log.
        is_debug: Whether the message should also be emitted to the local logger.

    Returns:
        None.
    """
    if is_debug:
        logger.info(message)

    if ctx:
        await ctx.info(message)
