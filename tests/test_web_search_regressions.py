"""
Regression tests for the web_search response cleanup behavior.
"""

import httpx
import pytest

from grok_search import server
from grok_search.providers.grok import (
    GrokSearchProvider,
    ResponsesUnsupportedError,
    _extract_responses_sources,
    _extract_responses_text,
    _join_api_url,
)
from grok_search.utils import sanitize_model_output, search_prompt


class _FakeStreamResponse:
    """
    Minimal async streaming response used for provider parsing tests.

    Args:
        lines: Streamed text lines exposed by aiter_lines.

    Returns:
        None.
    """

    def __init__(self, lines: list[str]) -> None:
        """
        Store the fake stream lines for later iteration.

        Args:
            lines: Streamed text lines exposed by aiter_lines.

        Returns:
            None.
        """
        self._lines = lines

    async def aiter_lines(self):
        """
        Yield the configured lines as an async iterator.

        Args:
            None.

        Returns:
            An async iterator over the configured lines.
        """
        for line in self._lines:
            yield line


class _FakeAsyncClient:
    """
    Minimal async HTTP client for provider and server transport tests.

    Args:
        post_responses: Ordered responses returned from POST calls.
        get_map: Mapping from URL to GET response or exception.

    Returns:
        None.
    """

    def __init__(
        self,
        *,
        post_responses: list[httpx.Response | Exception] | None = None,
        get_map: dict[str, httpx.Response | Exception] | None = None,
    ) -> None:
        """
        Store deterministic POST and GET behaviors for the fake client.

        Args:
            post_responses: Ordered responses returned from POST calls.
            get_map: Mapping from URL to GET response or exception.

        Returns:
            None.
        """
        self._post_responses = list(post_responses or [])
        self._get_map = dict(get_map or {})
        self.post_calls: list[dict[str, object]] = []
        self.get_calls: list[str] = []

    async def __aenter__(self):
        """
        Enter the async context manager.

        Args:
            None.

        Returns:
            The fake client instance.
        """
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        """
        Exit the async context manager.

        Args:
            exc_type: Exception type, if any.
            exc: Exception instance, if any.
            tb: Traceback, if any.

        Returns:
            False to propagate exceptions.
        """
        return False

    async def post(self, url: str, headers=None, json=None):
        """
        Return the next configured POST response.

        Args:
            url: The requested URL.
            headers: The request headers.
            json: The JSON payload.

        Returns:
            The configured HTTP response.
        """
        self.post_calls.append({"url": url, "headers": headers, "json": json})
        response_or_exc = self._post_responses.pop(0)
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc

    async def get(self, url: str, headers=None):
        """
        Return the configured GET response for the requested URL.

        Args:
            url: The requested URL.
            headers: The request headers.

        Returns:
            The configured HTTP response.
        """
        self.get_calls.append(url)
        response_or_exc = self._get_map[url]
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc


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
async def test_parse_streaming_response_accepts_message_content_payload():
    """
    Verify that the provider can parse non-delta message payloads.

    Args:
        None.

    Returns:
        None.
    """
    provider = GrokSearchProvider("https://example.invalid/v1", "test-key")
    response = _FakeStreamResponse(
        [
            'data: {"choices":[{"message":{"content":"Structured answer"}}]}',
            "data: [DONE]",
        ]
    )

    result = await provider._parse_streaming_response(response)

    assert result == "Structured answer"


def test_join_api_url_avoids_duplicate_v1():
    """
    Verify that endpoint joining keeps the version segment only once.

    Args:
        None.

    Returns:
        None.
    """
    assert (
        _join_api_url("https://example.invalid", "/responses")
        == "https://example.invalid/v1/responses"
    )
    assert (
        _join_api_url("https://example.invalid/v1", "/responses")
        == "https://example.invalid/v1/responses"
    )


def test_extract_responses_text_reads_output_text_blocks():
    """
    Verify that Responses message text is extracted from output_text blocks.

    Args:
        None.

    Returns:
        None.
    """
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "First line."},
                    {"type": "output_text", "text": "Second line."},
                ],
            }
        ]
    }

    assert _extract_responses_text(payload) == "First line.\nSecond line."


def test_extract_responses_sources_deduplicates_and_normalizes_titles():
    """
    Verify that Responses sources are deduplicated and numeric titles normalize.

    Args:
        None.

    Returns:
        None.
    """
    payload = {
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "sources": [
                        {
                            "title": "OpenAI Responses API",
                            "url": "https://platform.openai.com/docs/api-reference/responses",
                        }
                    ]
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Answer",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "1",
                                "url": "https://platform.openai.com/docs/api-reference/responses",
                            },
                            {
                                "type": "url_citation",
                                "title": "2",
                                "url": "https://developers.openai.com/api/docs/guides/migrate-to-responses/",
                            },
                        ],
                    }
                ],
            },
        ]
    }

    assert _extract_responses_sources(payload) == [
        {
            "title": "OpenAI Responses API",
            "url": "https://platform.openai.com/docs/api-reference/responses",
        },
        {
            "title": "developers.openai.com",
            "url": "https://developers.openai.com/api/docs/guides/migrate-to-responses/",
        },
    ]


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
        """
        Raise a deterministic upstream failure for the provider.

        Args:
            self: The provider instance.
            query: The search query.
            platform: The optional platform hint.
            min_results: The minimum requested result count.
            max_results: The maximum requested result count.
            ctx: Optional request context.

        Returns:
            None.

        Raises:
            RuntimeError: Always raised for the test.
        """
        raise RuntimeError("boom")

    monkeypatch.setattr(server.GrokSearchProvider, "search", boom)

    result = await server.web_search("What is the capital of France?")

    assert result["content"] == "Search upstream error: RuntimeError: boom"
    assert result["sources_count"] == 0


@pytest.mark.asyncio
async def test_search_falls_back_to_legacy_chat_when_responses_is_unsupported(
    monkeypatch,
):
    """
    Verify that unsupported Responses backends fall back to chat completions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    provider = GrokSearchProvider("https://example.invalid/v1", "test-key")

    async def unsupported(headers, payload, ctx=None):
        """
        Raise a deterministic unsupported-endpoint error.

        Args:
            headers: The request headers.
            payload: The Responses payload.
            ctx: Optional request context.

        Returns:
            None.
        """
        raise ResponsesUnsupportedError("unsupported")

    async def legacy(headers, payload, ctx=None):
        """
        Return the legacy fallback answer.

        Args:
            headers: The request headers.
            payload: The legacy chat-completions payload.
            ctx: Optional request context.

        Returns:
            The fallback answer text.
        """
        assert payload["messages"][0]["content"] == search_prompt
        return "Legacy answer\n\n## Sources\n- [FastMCP](https://gofastmcp.com)"

    monkeypatch.setattr(provider, "_execute_responses_search_with_retry", unsupported)
    monkeypatch.setattr(provider, "_execute_stream_with_retry", legacy)

    result = await provider.search("What is FastMCP?")

    assert result == "Legacy answer\n\n## Sources\n- [FastMCP](https://gofastmcp.com)"


@pytest.mark.asyncio
async def test_search_retries_responses_without_explicit_tools_for_auto_search_proxy(
    monkeypatch,
):
    """
    Verify that duplicate web_search tool errors retry Responses without tools.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    provider = GrokSearchProvider("https://example.invalid", "test-key")
    duplicate_tool_response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://example.invalid/v1/responses"),
        json={
            "error": {
                "message": "Multiple web search tools are not supported",
                "type": "bad_response_status_code",
            }
        },
    )
    success_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://example.invalid/v1/responses"),
        json={
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {
                                "title": "Example Source",
                                "url": "https://example.com/source",
                            }
                        ]
                    },
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Auto-search answer.",
                            "annotations": [],
                        }
                    ],
                },
            ]
        },
    )
    fake_client = _FakeAsyncClient(
        post_responses=[duplicate_tool_response, success_response]
    )

    async def legacy(headers, payload, ctx=None):
        """
        Fail the test if legacy chat fallback is used unexpectedly.

        Args:
            headers: The request headers.
            payload: The legacy payload.
            ctx: Optional request context.

        Returns:
            None.
        """
        raise AssertionError("Legacy chat fallback should not run")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: fake_client)
    monkeypatch.setattr(provider, "_execute_stream_with_retry", legacy)

    result = await provider.search("What is FastMCP?")

    assert result == (
        "Auto-search answer.\n\n## Sources\n- [Example Source](https://example.com/source)"
    )
    assert fake_client.post_calls[0]["json"]["tools"] == [{"type": "web_search"}]
    assert "tools" not in fake_client.post_calls[1]["json"]


@pytest.mark.asyncio
async def test_fetch_available_models_falls_back_to_v1_models(monkeypatch):
    """
    Verify that model probing falls back from /models to /v1/models.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    base_request = httpx.Request("GET", "https://example.invalid/models")
    v1_request = httpx.Request("GET", "https://example.invalid/v1/models")
    fake_client = _FakeAsyncClient(
        get_map={
            "https://example.invalid/models": httpx.Response(
                200,
                request=base_request,
                text="not-json",
            ),
            "https://example.invalid/v1/models": httpx.Response(
                200,
                request=v1_request,
                json={"data": [{"id": "grok-4.20-reasoning"}]},
            ),
        }
    )

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: fake_client)

    result = await server._fetch_available_models("https://example.invalid", "test-key")

    assert result == ["grok-4.20-reasoning"]
    assert fake_client.get_calls == [
        "https://example.invalid/models",
        "https://example.invalid/v1/models",
    ]


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
        """
        Return an empty provider response for fallback coverage.

        Args:
            self: The provider instance.
            query: The search query.
            platform: The optional platform hint.
            min_results: The minimum requested result count.
            max_results: The maximum requested result count.
            ctx: Optional request context.

        Returns:
            An empty string.
        """
        return ""

    async def tavily_results(query, max_results=6):
        """
        Return a deterministic Tavily search payload.

        Args:
            query: The search query.
            max_results: The maximum number of requested results.

        Returns:
            A single normalized Tavily result.
        """
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
        == "Search completed, but the upstream response did not contain "
        "a parsable answer body. Call get_sources to inspect the sources."
    )
    assert result["sources_count"] == 1


@pytest.mark.asyncio
async def test_web_search_retries_when_first_response_only_contains_sources(
    monkeypatch,
):
    """
    Verify that semantic retries recover from source-only upstream responses.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")

    calls = {"count": 0}

    async def flaky(self, query, platform="", min_results=3, max_results=10, ctx=None):
        """
        Return sources only on the first call and a full answer on retry.

        Args:
            self: The provider instance.
            query: The search query.
            platform: The optional platform hint.
            min_results: The minimum requested result count.
            max_results: The maximum requested result count.
            ctx: Optional request context.

        Returns:
            A simulated provider response string.
        """
        calls["count"] += 1
        if calls["count"] == 1:
            return "Sources\n- [FastMCP](https://gofastmcp.com)"
        return (
            "FastMCP is a framework for building MCP applications.\n\n"
            "Sources\n- [FastMCP](https://gofastmcp.com)"
        )

    monkeypatch.setattr(server.GrokSearchProvider, "search", flaky)

    result = await server.web_search("What is FastMCP?")

    assert calls["count"] == 2
    assert result["content"] == "FastMCP is a framework for building MCP applications."
    assert result["sources_count"] == 1
