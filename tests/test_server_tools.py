import json

import httpx
import pytest

from grok_search import server


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


@pytest.mark.asyncio
async def test_call_tavily_extract_retries_until_content_is_available(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_URL", "https://example.invalid")
    monkeypatch.setenv("GROK_RETRY_MAX_ATTEMPTS", "3")

    attempts = {"count": 0}
    sleeps: list[int] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, headers=None, json=None):
            attempts["count"] += 1
            if attempts["count"] < 3:
                return _FakeResponse({"results": [{"raw_content": "   "}]})
            return _FakeResponse({"results": [{"raw_content": "# Title"}]})

    async def fake_sleep(retry_index: int) -> None:
        sleeps.append(retry_index)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(server, "_sleep_before_retry", fake_sleep)

    result = await server._call_tavily_extract("https://example.com")

    assert result == "# Title"
    assert attempts["count"] == 3
    assert sleeps == [1, 2]


@pytest.mark.asyncio
async def test_switch_model_rejects_unknown_model(monkeypatch):
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(server.config, "_cached_model", "grok-4-fast", raising=False)

    saved: dict[str, str] = {}

    async def fake_fetch_available_models(api_url: str, api_key: str) -> list[str]:
        return ["grok-4-fast"]

    def fake_set_model(model: str) -> None:
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
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(server.config, "_cached_model", "grok-4-fast", raising=False)

    saved: dict[str, str] = {}

    async def fake_fetch_available_models(api_url: str, api_key: str) -> list[str]:
        return ["grok-4-fast", "grok-4.1-thing"]

    def fake_set_model(model: str) -> None:
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
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(server.config, "_cached_model", "grok-4-fast", raising=False)

    async def fake_describe(self, url: str, ctx=None) -> dict:
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
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(server.config, "_cached_model", "grok-4-fast", raising=False)

    async def fake_rank_sources(
        self, query: str, sources_text: str, total: int, ctx=None
    ) -> list[int]:
        return [2, 1]

    monkeypatch.setattr(server.GrokSearchProvider, "rank_sources", fake_rank_sources)

    result = await server.rank_sources("test query", "1. A\n2. B", 2)

    assert result == {"query": "test query", "order": [2, 1], "total": 2}
