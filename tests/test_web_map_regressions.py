"""
Regression tests for Tavily map request shaping and same-site filtering.
"""

import json

import httpx
import pytest

from grok_search import server


class _FakeResponse:
    """
    Minimal HTTP response stub used for map tool tests.

    Args:
        payload: JSON payload returned by the fake response.
        status_code: HTTP status code exposed by the response.
        text: Response text exposed on HTTP errors.

    Returns:
        None.
    """

    def __init__(self, payload: dict, status_code: int = 200, text: str = "") -> None:
        """
        Store payload and metadata for later assertions.

        Args:
            payload: JSON payload returned by the fake response.
            status_code: HTTP status code exposed by the response.
            text: Response text exposed on HTTP errors.

        Returns:
            None.
        """
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        """
        Simulate a successful HTTP response.

        Args:
            None.

        Returns:
            None.
        """
        return None

    def json(self) -> dict:
        """
        Return the configured JSON payload.

        Args:
            None.

        Returns:
            The configured JSON payload.
        """
        return self._payload


@pytest.mark.asyncio
async def test_call_tavily_map_requests_same_site_defaults_and_filters_results(
    monkeypatch,
):
    """
    Verify that map requests default to same-site crawling and filter external URLs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_URL", "https://example.invalid")

    captured: dict[str, object] = {}

    class FakeAsyncClient:
        """
        Fake async HTTP client for successful map requests.

        Args:
            None.

        Returns:
            None.
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
            return None

        async def __aenter__(self):
            """
            Enter the async context manager.

            Args:
                None.

            Returns:
                The fake client itself.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb):
            """
            Exit the async context manager.

            Args:
                exc_type: Exception type, if any.
                exc: Exception instance, if any.
                tb: Traceback object, if any.

            Returns:
                False to propagate exceptions.
            """
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
            return _FakeResponse(
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

    result = json.loads(
        await server._call_tavily_map(
            "https://example.com/docs",
            "only docs pages",
            max_depth=1,
            max_breadth=10,
            limit=10,
            timeout=30,
        )
    )

    assert captured["endpoint"] == "https://example.invalid/map"
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
async def test_call_tavily_map_falls_back_when_same_site_fields_are_rejected(
    monkeypatch,
):
    """
    Verify that legacy fallback still filters external results locally.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_URL", "https://example.invalid")

    payloads: list[dict] = []

    class FakeAsyncClient:
        """
        Fake async HTTP client that rejects new fields on the first request.

        Args:
            None.

        Returns:
            None.
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
            return None

        async def __aenter__(self):
            """
            Enter the async context manager.

            Args:
                None.

            Returns:
                The fake client itself.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb):
            """
            Exit the async context manager.

            Args:
                exc_type: Exception type, if any.
                exc: Exception instance, if any.
                tb: Traceback object, if any.

            Returns:
                False to propagate exceptions.
            """
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
            return _FakeResponse(
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

    result = json.loads(
        await server._call_tavily_map(
            "https://docs.python.org/3/library/asyncio.html",
            None,
            max_depth=1,
            max_breadth=8,
            limit=8,
            timeout=30,
        )
    )

    assert payloads[0]["allow_external"] is False
    assert payloads[0]["select_domains"] == ["docs.python.org"]
    assert "allow_external" not in payloads[1]
    assert "select_domains" not in payloads[1]
    assert result["results"] == ["https://docs.python.org/3/library/asyncio-task.html"]


@pytest.mark.asyncio
async def test_call_tavily_map_reports_filter_counts_when_all_results_are_removed(
    monkeypatch,
):
    """
    Verify that empty filtered map results expose raw and filtered counts.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_URL", "https://example.invalid")

    class FakeAsyncClient:
        """
        Fake async HTTP client for all-external map results.

        Args:
            None.

        Returns:
            None.
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
            return None

        async def __aenter__(self):
            """
            Enter the async context manager.

            Args:
                None.

            Returns:
                The fake client itself.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb):
            """
            Exit the async context manager.

            Args:
                exc_type: Exception type, if any.
                exc: Exception instance, if any.
                tb: Traceback object, if any.

            Returns:
                False to propagate exceptions.
            """
            return False

        async def post(self, endpoint, headers=None, json=None):
            """
            Return only external links for the requested site.

            Args:
                endpoint: The requested endpoint URL.
                headers: Outgoing request headers.
                json: Outgoing JSON request body.

            Returns:
                A fake successful response.
            """
            return _FakeResponse(
                {
                    "base_url": "https://gofastmcp.com",
                    "results": [
                        "https://modelcontextprotocol.io/",
                        "https://llmstxt.org/",
                    ],
                    "response_time": 0.8,
                }
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = json.loads(
        await server._call_tavily_map(
            "https://gofastmcp.com",
            "only documentation and getting-started pages",
            max_depth=1,
            max_breadth=10,
            limit=10,
            timeout=30,
        )
    )

    assert result["results"] == []
    assert result["raw_count"] == 2
    assert result["filtered_out_count"] == 2


@pytest.mark.asyncio
async def test_call_tavily_map_probes_legacy_results_when_constrained_response_is_empty(
    monkeypatch,
):
    """
    Verify that empty constrained results can expose counts via a legacy probe.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_URL", "https://example.invalid")

    payloads: list[dict] = []

    class FakeAsyncClient:
        """
        Fake async HTTP client for the empty-then-legacy-probe path.

        Args:
            None.

        Returns:
            None.
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
            return None

        async def __aenter__(self):
            """
            Enter the async context manager.

            Args:
                None.

            Returns:
                The fake client itself.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb):
            """
            Exit the async context manager.

            Args:
                exc_type: Exception type, if any.
                exc: Exception instance, if any.
                tb: Traceback object, if any.

            Returns:
                False to propagate exceptions.
            """
            return False

        async def post(self, endpoint, headers=None, json=None):
            """
            Return an empty constrained result followed by external legacy data.

            Args:
                endpoint: The requested endpoint URL.
                headers: Outgoing request headers.
                json: Outgoing JSON request body.

            Returns:
                A fake successful response.
            """
            payloads.append(json)
            if len(payloads) == 1:
                return _FakeResponse(
                    {
                        "base_url": "https://gofastmcp.com",
                        "results": [],
                        "response_time": 0.8,
                    }
                )
            return _FakeResponse(
                {
                    "base_url": "https://gofastmcp.com",
                    "results": [
                        "https://llmstxt.org/",
                        "https://modelcontextprotocol.io/",
                    ],
                    "response_time": 1.2,
                }
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = json.loads(
        await server._call_tavily_map(
            "https://gofastmcp.com",
            "only documentation and getting-started pages",
            max_depth=1,
            max_breadth=10,
            limit=10,
            timeout=30,
        )
    )

    assert payloads[0]["allow_external"] is False
    assert payloads[0]["select_domains"] == ["gofastmcp.com"]
    assert "allow_external" not in payloads[1]
    assert "select_domains" not in payloads[1]
    assert result["results"] == []
    assert result["raw_count"] == 2
    assert result["filtered_out_count"] == 2
