"""Regression tests for server helpers that expose provider and config behavior."""

import json

import httpx
import pytest

from grok_search import server


class _FakeResponse:
    """
    Provide a minimal JSON response stub for server helper tests.

    Args:
        payload: The JSON payload returned by the fake response.

    Returns:
        None.
    """

    def __init__(self, payload: dict):
        """
        Store the fake response payload.

        Args:
            payload: The JSON payload returned by the fake response.

        Returns:
            None.
        """
        self._payload = payload

    def raise_for_status(self) -> None:
        """
        Simulate a successful HTTP response.

        Returns:
            None.
        """
        return None

    def json(self) -> dict:
        """
        Return the configured JSON payload.

        Returns:
            The fake JSON payload.
        """
        return self._payload


@pytest.mark.asyncio
async def test_call_tavily_extract_retries_until_content_is_available(monkeypatch):
    """
    Verify that Tavily extract retries until non-empty content appears.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_URL", "https://example.invalid")
    monkeypatch.setenv("GROK_RETRY_MAX_ATTEMPTS", "3")

    attempts = {"count": 0}
    sleeps: list[int] = []

    class FakeAsyncClient:
        """
        Provide a fake async client for Tavily extract retries.

        Returns:
            None.
        """

        def __init__(self, *args, **kwargs):
            """
            Ignore httpx.AsyncClient constructor arguments.

            Args:
                *args: Positional constructor arguments.
                **kwargs: Keyword constructor arguments.

            Returns:
                None.
            """
            pass

        async def __aenter__(self):
            """
            Enter the async context manager.

            Returns:
                The fake client instance.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb):
            """
            Exit the async context manager.

            Args:
                exc_type: The exception type, if any.
                exc: The exception instance, if any.
                tb: The traceback object, if any.

            Returns:
                False to propagate exceptions.
            """
            return False

        async def post(self, endpoint, headers=None, json=None):
            """
            Return blank content before eventually returning extracted Markdown.

            Args:
                endpoint: The requested endpoint URL.
                headers: Outgoing request headers.
                json: Outgoing request body.

            Returns:
                A fake Tavily extract response.
            """
            attempts["count"] += 1
            if attempts["count"] < 3:
                return _FakeResponse({"results": [{"raw_content": "   "}]})
            return _FakeResponse({"results": [{"raw_content": "# Title"}]})

    async def fake_sleep(retry_index: int) -> None:
        """
        Record retry delays instead of sleeping.

        Args:
            retry_index: The 1-based retry attempt index.

        Returns:
            None.
        """
        sleeps.append(retry_index)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(server, "_sleep_before_retry", fake_sleep)

    result = await server._call_tavily_extract("https://example.com")

    assert result == "# Title"
    assert attempts["count"] == 3
    assert sleeps == [1, 2]


@pytest.mark.asyncio
async def test_switch_model_rejects_unknown_model(monkeypatch):
    """
    Verify that unknown models are rejected before persistence.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(server.config, "_cached_model", "grok-4-fast", raising=False)

    saved: dict[str, str] = {}

    async def fake_fetch_available_models(api_url: str, api_key: str) -> list[str]:
        """
        Return a single available model for validation.

        Args:
            api_url: The upstream API URL.
            api_key: The upstream API key.

        Returns:
            A list containing the only valid model.
        """
        return ["grok-4-fast"]

    def fake_set_model(model: str) -> None:
        """
        Record the model that would have been persisted.

        Args:
            model: The model identifier to persist.

        Returns:
            None.
        """
        saved["model"] = model
        server.config._cached_model = model

    monkeypatch.setattr(server, "_fetch_available_models", fake_fetch_available_models)
    monkeypatch.setattr(server.config, "set_model", fake_set_model)

    result = json.loads(await server.switch_model("grok-4.1-thing"))

    assert result["status"] == "failed"
    assert "Invalid model: grok-4.1-thing" in result["message"]
    assert "model" not in saved


@pytest.mark.asyncio
async def test_switch_model_accepts_validated_model(monkeypatch):
    """
    Verify that validated models are persisted successfully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(server.config, "_cached_model", "grok-4-fast", raising=False)

    saved: dict[str, str] = {}

    async def fake_fetch_available_models(api_url: str, api_key: str) -> list[str]:
        """
        Return multiple valid models for validation.

        Args:
            api_url: The upstream API URL.
            api_key: The upstream API key.

        Returns:
            The list of valid model identifiers.
        """
        return ["grok-4-fast", "grok-4.1-thing"]

    def fake_set_model(model: str) -> None:
        """
        Record the validated model that would have been persisted.

        Args:
            model: The model identifier to persist.

        Returns:
            None.
        """
        saved["model"] = model
        server.config._cached_model = model

    monkeypatch.setattr(server, "_fetch_available_models", fake_fetch_available_models)
    monkeypatch.setattr(server.config, "set_model", fake_set_model)

    result = json.loads(await server.switch_model("grok-4.1-thing"))

    assert result["status"] == "success"
    assert result["current_model"] == "grok-4.1-thing"
    assert saved["model"] == "grok-4.1-thing"


@pytest.mark.asyncio
async def test_describe_url_tool_exposes_provider(monkeypatch):
    """
    Verify that the describe_url tool delegates to the provider.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(server.config, "_cached_model", "grok-4-fast", raising=False)

    async def fake_describe(self, url: str, ctx=None) -> dict:
        """
        Return a deterministic provider description payload.

        Args:
            self: The provider instance.
            url: The described URL.
            ctx: Optional request context.

        Returns:
            A fixed description payload.
        """
        return {"title": "Example", "extracts": '"alpha" | "beta"', "url": url}

    monkeypatch.setattr(server.GrokSearchProvider, "describe_url", fake_describe)

    result = await server.describe_url("https://example.com")

    assert result == {
        "title": "Example",
        "extracts": '"alpha" | "beta"',
        "url": "https://example.com",
    }


@pytest.mark.asyncio
async def test_rank_sources_canonicalizes_markdown_input_before_provider(monkeypatch):
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(server.config, "_cached_model", "grok-4-fast", raising=False)
    captured: dict[str, object] = {}

    async def fake_rank_sources(
        self, query: str, sources_text: str, total: int, ctx=None
    ) -> list[int]:
        captured["sources_text"] = sources_text
        captured["total"] = total
        return [2, 1]

    monkeypatch.setattr(server.GrokSearchProvider, "rank_sources", fake_rank_sources)

    result = await server.rank_sources(
        "test query",
        "1. [Alpha]( https://example.com/a )\n"
        "2. [Alpha duplicate](https://example.com/a)\n"
        "3. [Beta](https://example.com/b)",
        3,
    )

    assert captured == {
        "sources_text": "1. [Alpha](https://example.com/a)\n2. [Beta](https://example.com/b)",
        "total": 2,
    }
    assert result == {"query": "test query", "order": [2, 1], "total": 2}


@pytest.mark.asyncio
async def test_rank_sources_tool_exposes_provider(monkeypatch):
    """
    Verify that the rank_sources tool delegates to the provider.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(server.config, "_cached_model", "grok-4-fast", raising=False)

    async def fake_rank_sources(
        self, query: str, sources_text: str, total: int, ctx=None
    ) -> list[int]:
        """
        Return a deterministic ranking from the fake provider.

        Args:
            self: The provider instance.
            query: The ranking query.
            sources_text: The numbered source list.
            total: The number of sources.
            ctx: Optional request context.

        Returns:
            The deterministic ranked order.
        """
        return [2, 1]

    monkeypatch.setattr(server.GrokSearchProvider, "rank_sources", fake_rank_sources)

    result = await server.rank_sources("test query", "1. A\n2. B", 2)
    assert result == {"query": "test query", "order": [2, 1], "total": 2}


def test_build_tavily_map_body_adds_same_site_scope_for_scheme_less_urls():
    """
    Verify that map requests scope bare domain inputs the way Tavily examples allow.

    Args:
        None.

    Returns:
        None.
    """
    body = server._build_tavily_map_body(
        "docs.example.com",
        instructions="only API reference pages",
        max_depth=2,
        max_breadth=5,
        limit=10,
        timeout=30,
    )

    assert body["url"] == "docs.example.com"
    assert body["instructions"] == "only API reference pages"
    assert body["max_depth"] == 2
    assert body["select_domains"] == ["docs.example.com"]
    assert body["allow_external"] is False


@pytest.mark.asyncio
async def test_call_tavily_map_uses_legacy_same_site_results_when_strict_scope_empty(
    monkeypatch,
):
    """
    Verify that an unscoped retry can recover same-site URLs from an empty strict map.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_URL", "https://example.invalid")

    post_calls: list[dict[str, object]] = []
    responses = [
        {"base_url": "https://docs.example.com", "results": [], "response_time": 0.1},
        {
            "base_url": "https://docs.example.com",
            "results": [
                "https://docs.example.com/guide",
                "https://external.example.net/nope",
                {"url": "https://www.docs.example.com/api", "title": "API"},
            ],
            "response_time": 0.2,
        },
    ]

    class FakeAsyncClient:
        """
        Provide deterministic Tavily map responses for strict and legacy calls.

        Returns:
            None.
        """

        def __init__(self, *args, **kwargs):
            """
            Ignore httpx.AsyncClient constructor arguments.

            Args:
                *args: Positional constructor arguments.
                **kwargs: Keyword constructor arguments.

            Returns:
                None.
            """
            pass

        async def __aenter__(self):
            """
            Enter the async context manager.

            Returns:
                The fake client instance.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb):
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

        async def post(self, endpoint, headers=None, json=None):
            """
            Return the next configured map response and record the outgoing body.

            Args:
                endpoint: The requested endpoint URL.
                headers: Outgoing request headers.
                json: Outgoing request body.

            Returns:
                A fake Tavily response.
            """
            post_calls.append({"endpoint": endpoint, "headers": headers, "json": json})
            return _FakeResponse(responses.pop(0))

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = json.loads(
        await server._call_tavily_map(
            "https://docs.example.com",
            instructions="only docs pages",
            max_depth=2,
            max_breadth=5,
            limit=10,
            timeout=30,
        )
    )

    assert result["results"] == [
        "https://docs.example.com/guide",
        {"url": "https://www.docs.example.com/api", "title": "API"},
    ]
    assert result["raw_count"] == 3
    assert result["filtered_out_count"] == 1
    assert result["fallback_used"] == "legacy_unscoped_request"
    assert post_calls[0]["json"] == {
        "url": "https://docs.example.com",
        "instructions": "only docs pages",
        "max_depth": 2,
        "max_breadth": 5,
        "limit": 10,
        "timeout": 30,
        "allow_external": False,
        "select_domains": ["docs.example.com"],
    }
    assert post_calls[1]["json"] == {
        "url": "https://docs.example.com",
        "instructions": "only docs pages",
        "max_depth": 2,
        "max_breadth": 5,
        "limit": 10,
        "timeout": 30,
    }
