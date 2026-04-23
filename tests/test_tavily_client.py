"""Regression tests for Tavily client request shaping and filtering."""

import httpx
import pytest

from grok_search.clients.tavily import TavilyClient, normalize_tavily_api_base_url
from grok_search.config import config


class FakeResponse:
    """
    Minimal HTTP response stub used for Tavily client tests.

    Attributes:
        status_code: The fake HTTP status code.
        text: The fake response body text.
    """

    def __init__(self, payload: dict, status_code: int = 200, text: str = "") -> None:
        """
        Store payload and metadata for later assertions.

        Args:
            payload: The JSON payload returned by the fake response.
            status_code: The fake HTTP status code.
            text: The fake HTTP response body text.

        Returns:
            None.
        """
        self._payload = payload
        self.status_code = status_code
        self.text = text

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
            The configured JSON payload.
        """
        return self._payload


def test_normalize_tavily_api_base_url_accepts_hikari_mcp_url() -> None:
    """
    Verify that Hikari MCP URLs are converted to the Tavily HTTP facade.

    Returns:
        None.
    """
    assert (
        normalize_tavily_api_base_url("https://tavily.example.com/mcp")
        == "https://tavily.example.com/api/tavily"
    )
    assert (
        normalize_tavily_api_base_url("https://proxy.example.com/prefix/mcp/")
        == "https://proxy.example.com/prefix/api/tavily"
    )
    assert (
        normalize_tavily_api_base_url("https://api.tavily.com/")
        == "https://api.tavily.com"
    )


@pytest.mark.asyncio
async def test_map_site_requests_same_site_defaults_and_filters_results(
    monkeypatch,
) -> None:
    """
    Verify that map requests default to same-site crawling and filter external URLs.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_URL", "https://example.invalid/mcp")

    captured: dict[str, object] = {}

    class FakeAsyncClient:
        """
        Fake async HTTP client for successful map requests.
        """

        def __init__(self, *args, **kwargs) -> None:
            """
            Ignore constructor arguments from httpx.AsyncClient.

            Args:
                *args: Positional constructor arguments.
                **kwargs: Keyword constructor arguments.

            Returns:
                None.
            """
            del args, kwargs

        async def __aenter__(self):
            """
            Enter the async context manager.

            Returns:
                The fake client itself.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            """
            Exit the async context manager.

            Args:
                exc_type: The exception type, if any.
                exc: The exception instance, if any.
                tb: The traceback object, if any.

            Returns:
                False to propagate exceptions.
            """
            del exc_type, exc, tb
            return False

        async def post(self, endpoint, headers=None, json=None):
            """
            Record the outgoing payload and return a mixed-site response.

            Args:
                endpoint: The requested endpoint URL.
                headers: Outgoing request headers.
                json: Outgoing JSON request body.

            Returns:
                A fake successful response.
            """
            captured["endpoint"] = endpoint
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse(
                {
                    "base_url": "https://example.com/docs",
                    "results": [
                        "https://example.com/docs/page-a",
                        "https://www.example.com/docs/page-b",
                        "https://external.example.net/",
                    ],
                    "response_time": 1.5,
                }
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = await TavilyClient(config).map_site(
        "https://example.com/docs",
        "only docs pages",
        max_depth=1,
        max_breadth=10,
        limit=10,
        timeout=30,
    )

    assert captured["endpoint"] == "https://example.invalid/api/tavily/map"
    assert captured["json"] == {
        "url": "https://example.com/docs",
        "max_depth": 1,
        "max_breadth": 10,
        "limit": 10,
        "timeout": 30,
        "instructions": "only docs pages",
        "allow_external": False,
        "select_domains": ["example.com"],
    }
    assert result["results"] == [
        "https://example.com/docs/page-a",
        "https://www.example.com/docs/page-b",
    ]


@pytest.mark.asyncio
async def test_map_site_falls_back_when_same_site_fields_are_rejected(
    monkeypatch,
) -> None:
    """
    Verify that legacy fallback still filters external results locally.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_URL", "https://example.invalid")

    payloads: list[dict] = []

    class FakeAsyncClient:
        """
        Fake async HTTP client that rejects new fields on the first request.
        """

        def __init__(self, *args, **kwargs) -> None:
            """
            Ignore constructor arguments from httpx.AsyncClient.

            Args:
                *args: Positional constructor arguments.
                **kwargs: Keyword constructor arguments.

            Returns:
                None.
            """
            del args, kwargs

        async def __aenter__(self):
            """
            Enter the async context manager.

            Returns:
                The fake client itself.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            """
            Exit the async context manager.

            Args:
                exc_type: The exception type, if any.
                exc: The exception instance, if any.
                tb: The traceback object, if any.

            Returns:
                False to propagate exceptions.
            """
            del exc_type, exc, tb
            return False

        async def post(self, endpoint, headers=None, json=None):
            """
            Reject the first same-site payload, then return a successful response.

            Args:
                endpoint: The requested endpoint URL.
                headers: Outgoing request headers.
                json: Outgoing JSON request body.

            Returns:
                A fake response or a raised HTTPStatusError.
            """
            del headers
            payloads.append(json)
            if len(payloads) == 1:
                request = httpx.Request("POST", endpoint)
                response = httpx.Response(
                    400,
                    request=request,
                    text="unknown field allow_external",
                )
                raise httpx.HTTPStatusError(
                    "Bad Request", request=request, response=response
                )
            return FakeResponse(
                {
                    "base_url": "https://docs.python.org/3/library/asyncio.html",
                    "results": [
                        "https://docs.python.org/3/library/asyncio-task.html",
                        "https://modelcontextprotocol.io/",
                    ],
                    "response_time": 1.7,
                }
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = await TavilyClient(config).map_site(
        "https://docs.python.org/3/library/asyncio.html",
        None,
        max_depth=1,
        max_breadth=8,
        limit=8,
        timeout=30,
    )

    assert payloads[0]["allow_external"] is False
    assert payloads[0]["select_domains"] == ["docs.python.org"]
    assert "allow_external" not in payloads[1]
    assert "select_domains" not in payloads[1]
    assert result["results"] == ["https://docs.python.org/3/library/asyncio-task.html"]
