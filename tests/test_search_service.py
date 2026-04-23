"""Regression tests for the search orchestration service."""

import pytest

from grok_search.config import config
from grok_search.services.search import SearchService


class FakeGrokService:
    """
    Minimal Grok service stub for search service tests.

    Attributes:
        responses: The queued responses or exceptions returned by search calls.
        calls: The number of executed search calls.
    """

    def __init__(self, responses: list[object]) -> None:
        """
        Store the queued fake search responses.

        Args:
            responses: The queued responses or exceptions returned by search calls.

        Returns:
            None.
        """
        self.responses = responses
        self.calls = 0

    async def search(self, query: str, platform: str = "", model: str = "") -> str:
        """
        Return the next queued fake search response.

        Args:
            query: The search query.
            platform: The optional platform hint.
            model: The optional model override.

        Returns:
            The queued response string.

        Raises:
            Exception: Raised when the queued response is an exception.
        """
        del query, platform, model
        self.calls += 1
        response = self.responses[self.calls - 1]
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, str)
        return response


class FakeTavilyClient:
    """
    Minimal Tavily client stub for search service tests.

    Attributes:
        is_configured: Whether Tavily should be treated as configured.
        results: The queued Tavily search results.
    """

    def __init__(self, is_configured: bool, results: list[dict] | None = None) -> None:
        """
        Store fake Tavily search state.

        Args:
            is_configured: Whether Tavily should be treated as configured.
            results: The fake Tavily search results.

        Returns:
            None.
        """
        self.is_configured = is_configured
        self.results = results

    async def search(self, query: str, max_results: int = 6) -> list[dict] | None:
        """
        Return the configured fake Tavily search results.

        Args:
            query: The search query.
            max_results: The maximum requested result count.

        Returns:
            The fake Tavily search results.
        """
        del query, max_results
        return self.results


@pytest.mark.asyncio
async def test_web_search_surfaces_upstream_errors(monkeypatch) -> None:
    """
    Verify that upstream Grok failures are surfaced in the search response.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")

    search_service = SearchService(
        config=config,
        grok_service=FakeGrokService([RuntimeError("boom")]),
        tavily_client=FakeTavilyClient(is_configured=False),
    )

    result = await search_service.web_search("What is the capital of France?")

    assert result["content"] == "Search upstream error: RuntimeError: boom"
    assert result["sources_count"] == 0


@pytest.mark.asyncio
async def test_web_search_reports_missing_body_when_only_sources_exist(
    monkeypatch,
) -> None:
    """
    Verify that source-only results produce an explicit fallback message.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")

    search_service = SearchService(
        config=config,
        grok_service=FakeGrokService(["", ""]),
        tavily_client=FakeTavilyClient(
            is_configured=True,
            results=[
                {
                    "title": "OpenAI Docs",
                    "url": "https://developers.openai.com/api/docs/guides/migrate-to-responses/",
                    "content": "Official migration guide",
                }
            ],
        ),
    )

    result = await search_service.web_search(
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
) -> None:
    """
    Verify that semantic retries recover from source-only upstream responses.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")

    grok_service = FakeGrokService(
        [
            "Sources\n- [FastMCP](https://gofastmcp.com)",
            (
                "FastMCP is a framework for building MCP applications.\n\n"
                "Sources\n- [FastMCP](https://gofastmcp.com)"
            ),
        ]
    )
    search_service = SearchService(
        config=config,
        grok_service=grok_service,
        tavily_client=FakeTavilyClient(is_configured=False),
    )

    result = await search_service.web_search("What is FastMCP?")

    assert grok_service.calls == 2
    assert result["content"] == "FastMCP is a framework for building MCP applications."
    assert result["sources_count"] == 1


@pytest.mark.asyncio
async def test_get_sources_returns_expiration_error_for_unknown_session() -> None:
    """
    Verify that unknown source sessions return the expected expiration payload.

    Returns:
        None.
    """
    search_service = SearchService(
        config=config,
        grok_service=FakeGrokService(["unused"]),
        tavily_client=FakeTavilyClient(is_configured=False),
    )

    result = await search_service.get_sources("missing-session")

    assert result == {
        "session_id": "missing-session",
        "sources": [],
        "sources_count": 0,
        "error": "session_id_not_found_or_expired",
    }
