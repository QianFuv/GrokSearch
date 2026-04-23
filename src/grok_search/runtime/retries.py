"""Retry configuration and delay helpers."""

import asyncio
from dataclasses import dataclass

from ..config import Config


@dataclass(frozen=True)
class RetrySettings:
    """
    Describe retry timing parameters used by upstream calls.

    Attributes:
        max_attempts: The maximum number of retry attempts.
        multiplier: The exponential backoff multiplier.
        max_wait: The retry backoff cap in seconds.
    """

    max_attempts: int
    multiplier: float
    max_wait: int


def get_retry_settings(config: Config) -> RetrySettings:
    """
    Build retry settings from the active configuration.

    Args:
        config: The runtime configuration instance.

    Returns:
        The retry settings derived from the configuration.
    """
    return RetrySettings(
        max_attempts=max(1, config.retry_max_attempts),
        multiplier=config.retry_multiplier,
        max_wait=config.retry_max_wait,
    )


def get_retry_delay_seconds(retry_index: int, settings: RetrySettings) -> float:
    """
    Compute the retry delay for a retry attempt index.

    Args:
        retry_index: The 1-based retry attempt index.
        settings: The retry timing settings.

    Returns:
        The retry delay in seconds.
    """
    delay = settings.multiplier * (2 ** max(retry_index - 1, 0))
    return min(float(settings.max_wait), max(0.0, delay))


async def sleep_before_retry(retry_index: int, settings: RetrySettings) -> None:
    """
    Sleep for the configured retry delay when positive.

    Args:
        retry_index: The 1-based retry attempt index.
        settings: The retry timing settings.

    Returns:
        None.
    """
    delay = get_retry_delay_seconds(retry_index, settings)
    if delay > 0:
        await asyncio.sleep(delay)
