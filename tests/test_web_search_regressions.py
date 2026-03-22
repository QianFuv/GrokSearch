"""
Regression tests for the web_search response cleanup behavior.
"""

import pytest

from grok_search import server
from grok_search.utils import sanitize_model_output


def test_sanitize_model_output_removes_think_and_meta_refusal():
    """
    Verify that hidden reasoning and meta refusals are stripped.

    Args:
        None.

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
async def test_web_search_surfaces_upstream_errors(monkeypatch):
    """
    Verify that upstream failures are surfaced in the tool response.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")

    async def boom(self, query, platform="", min_results=3, max_results=10, ctx=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(server.GrokSearchProvider, "search", boom)

    result = await server.web_search("What is the capital of France?")

    assert result["content"] == "搜索上游异常: RuntimeError: boom"
    assert result["sources_count"] == 0


@pytest.mark.asyncio
async def test_web_search_reports_missing_body_when_only_sources_exist(monkeypatch):
    """
    Verify that source-only results produce an explicit fallback message.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    async def empty(self, query, platform="", min_results=3, max_results=10, ctx=None):
        return ""

    async def tavily_results(query, max_results=6):
        return [
            {
                "title": "OpenAI Docs",
                "url": "https://developers.openai.com/api/docs/guides/migrate-to-responses/",
                "content": "Official migration guide",
            }
        ]

    monkeypatch.setattr(server.GrokSearchProvider, "search", empty)
    monkeypatch.setattr(server, "_call_tavily_search", tavily_results)

    result = await server.web_search(
        "Find the official OpenAI Responses API documentation.",
        extra_sources=1,
    )

    assert (
        result["content"]
        == "搜索完成，但上游未返回可解析正文。可调用 get_sources 查看来源。"
    )
    assert result["sources_count"] == 1
