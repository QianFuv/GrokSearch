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
