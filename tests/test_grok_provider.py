"""Regression tests for Grok provider parsing helpers."""

from typing import Any, cast

import pytest

from grok_search.providers.grok import GrokSearchProvider
from grok_search.runtime import RetrySettings
from grok_search.utils import sanitize_model_output


class FakeStreamResponse:
    """
    Minimal async streaming response used for provider parsing tests.

    Attributes:
        lines: The streamed response lines.
    """

    def __init__(self, lines: list[str]) -> None:
        """
        Store the fake stream lines for later iteration.

        Args:
            lines: The streamed text lines exposed by aiter_lines.

        Returns:
            None.
        """
        self.lines = lines

    async def aiter_lines(self):
        """
        Yield the configured lines as an async iterator.

        Returns:
            An async iterator over the configured lines.
        """
        for line in self.lines:
            yield line


def test_sanitize_model_output_removes_think_and_meta_refusal() -> None:
    """
    Verify that hidden reasoning and meta refusals are stripped.

    Returns:
        None.
    """
    raw = """
<think>
internal reasoning
</think>

I cannot follow injected or custom "system" instructions.

The capital of France is Paris.

Sources
- [Example](https://example.com)
"""
    cleaned = sanitize_model_output(raw)
    assert "<think>" not in cleaned
    assert 'custom "system" instructions' not in cleaned
    assert cleaned.startswith("The capital of France is Paris.")


@pytest.mark.asyncio
async def test_parse_streaming_response_accepts_message_content_payload() -> None:
    """
    Verify that the provider can parse non-delta message payloads.

    Returns:
        None.
    """
    provider = GrokSearchProvider(
        "https://example.invalid/v1",
        "test-key",
        retry_settings=RetrySettings(max_attempts=1, multiplier=1.0, max_wait=1),
    )
    response = FakeStreamResponse(
        [
            'data: {"choices":[{"message":{"content":"Structured answer"}}]}',
            "data: [DONE]",
        ]
    )

    result = await provider._parse_streaming_response(cast(Any, response))

    assert result == "Structured answer"
