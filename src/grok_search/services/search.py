"""Search orchestration service built on Grok and Tavily clients."""

import asyncio
from typing import Any, Protocol

from ..config import Config
from ..runtime import get_retry_settings, sleep_before_retry
from ..sources import (
    SourcesCache,
    merge_sources,
    new_session_id,
    split_answer_and_sources,
)


class GrokSearchProtocol(Protocol):
    """
    Describe the Grok search behavior required by the search service.
    """

    async def search(self, query: str, platform: str = "", model: str = "") -> str:
        """
        Execute a Grok-backed search request.

        Args:
            query: The search query.
            platform: The optional platform hint.
            model: The optional model override.

        Returns:
            The raw search response text.
        """


class TavilySearchProtocol(Protocol):
    """
    Describe the Tavily search behavior required by the search service.
    """

    @property
    def is_configured(self) -> bool:
        """
        Report whether Tavily is configured for use.

        Returns:
            True when Tavily is configured.
        """

    async def search(self, query: str, max_results: int = 6) -> list[dict] | None:
        """
        Execute a Tavily search request.

        Args:
            query: The search query.
            max_results: The maximum requested result count.

        Returns:
            The optional Tavily search results.
        """


def extra_results_to_sources(tavily_results: list[dict] | None) -> list[dict]:
    """
    Convert Tavily search results into normalized source records.

    Args:
        tavily_results: Raw Tavily search results.

    Returns:
        A normalized list of source dictionaries.
    """
    sources: list[dict] = []
    seen: set[str] = set()

    if tavily_results:
        for result in tavily_results:
            url = (result.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            source_item: dict = {"url": url, "provider": "tavily"}
            title = (result.get("title") or "").strip()
            if title:
                source_item["title"] = title
            content = (result.get("content") or "").strip()
            if content:
                source_item["description"] = content
            sources.append(source_item)

    return sources


class SearchService:
    """
    Coordinate Grok search, optional Tavily enrichment, and source caching.

    Attributes:
        _config: The runtime configuration instance.
        _grok_service: The Grok service used for upstream requests.
        _tavily_client: The Tavily client used for optional enrichment.
        _sources_cache: The in-memory search source cache.
    """

    def __init__(
        self,
        config: Config,
        grok_service: GrokSearchProtocol,
        tavily_client: TavilySearchProtocol,
    ) -> None:
        """
        Initialize the search service.

        Args:
            config: The runtime configuration instance.
            grok_service: The Grok service used for upstream requests.
            tavily_client: The Tavily client used for optional enrichment.

        Returns:
            None.
        """
        self._config = config
        self._grok_service = grok_service
        self._tavily_client = tavily_client
        self._sources_cache = SourcesCache(max_size=256)

    async def _search_with_answer_retry(
        self,
        query: str,
        platform: str,
        model: str,
        max_attempts: int = 2,
    ) -> str:
        """
        Retry a Grok search when the upstream response has no usable answer body.

        Args:
            query: The search query text.
            platform: Optional platform focus hint.
            model: An optional per-request model override.
            max_attempts: The maximum number of semantic retries.

        Returns:
            The final raw model response text.
        """
        final_result = ""
        total_attempts = max(1, max_attempts)
        retry_settings = get_retry_settings(self._config)

        for attempt in range(1, total_attempts + 1):
            final_result = await self._grok_service.search(query, platform, model)
            answer, _ = split_answer_and_sources(final_result)
            if answer.strip():
                return final_result
            if attempt < total_attempts:
                await sleep_before_retry(attempt, retry_settings)

        return final_result

    async def web_search(
        self,
        query: str,
        platform: str = "",
        model: str = "",
        extra_sources: int = 0,
    ) -> dict[str, Any]:
        """
        Execute a Grok web search and cache any extracted sources.

        Args:
            query: The natural-language search query.
            platform: An optional platform focus hint.
            model: An optional per-request model override.
            extra_sources: The number of extra Tavily sources to request.

        Returns:
            A dictionary containing the session ID, answer text, and source count.
        """
        session_id = new_session_id()
        try:
            _ = self._config.grok_api_url
            _ = self._config.grok_api_key
        except ValueError as error:
            await self._sources_cache.set(session_id, [])
            return {
                "session_id": session_id,
                "content": f"Configuration error: {error}",
                "sources_count": 0,
            }

        tavily_count = (
            extra_sources
            if extra_sources > 0 and self._tavily_client.is_configured
            else 0
        )

        async def safe_grok() -> str:
            """
            Execute the Grok search path with semantic answer retries.

            Returns:
                The raw Grok response text.
            """
            return await self._search_with_answer_retry(
                query=query,
                platform=platform,
                model=model,
            )

        async def safe_tavily() -> list[dict] | None:
            """
            Fetch extra Tavily sources while swallowing optional failures.

            Returns:
                A list of Tavily results, or None when unavailable.
            """
            try:
                if tavily_count:
                    return await self._tavily_client.search(query, tavily_count)
            except Exception:
                return None
            return None

        try:
            coroutines: list = [safe_grok()]
            if tavily_count > 0:
                coroutines.append(safe_tavily())
            gathered = list(await asyncio.gather(*coroutines, return_exceptions=True))
        except ValueError as error:
            await self._sources_cache.set(session_id, [])
            return {
                "session_id": session_id,
                "content": str(error),
                "sources_count": 0,
            }

        grok_first = gathered[0]
        grok_error: Exception | None = (
            grok_first if isinstance(grok_first, Exception) else None
        )
        grok_result = grok_first if isinstance(grok_first, str) else ""
        tavily_results: list[dict] | None = None
        if len(gathered) > 1:
            tavily_value = gathered[1]
            tavily_results = tavily_value if isinstance(tavily_value, list) else None

        answer, grok_sources = split_answer_and_sources(grok_result)
        extra = extra_results_to_sources(tavily_results)
        all_sources = merge_sources(grok_sources, extra)

        if not answer:
            if grok_error is not None:
                answer = (
                    f"Search upstream error: {type(grok_error).__name__}: {grok_error}"
                )
            elif all_sources:
                answer = (
                    "Search completed, but the upstream response did not contain a "
                    "parsable answer body. Call get_sources to inspect the sources."
                )
            else:
                answer = (
                    "Search completed, but the upstream response was empty after "
                    "retries."
                )

        await self._sources_cache.set(session_id, all_sources)
        return {
            "session_id": session_id,
            "content": answer,
            "sources_count": len(all_sources),
        }

    async def get_sources(self, session_id: str) -> dict[str, Any]:
        """
        Retrieve cached sources for a previous web_search session.

        Args:
            session_id: The session identifier returned by web_search.

        Returns:
            A dictionary containing the cached sources or an expiration error.
        """
        sources = await self._sources_cache.get(session_id)
        if sources is None:
            return {
                "session_id": session_id,
                "sources": [],
                "sources_count": 0,
                "error": "session_id_not_found_or_expired",
            }
        return {
            "session_id": session_id,
            "sources": sources,
            "sources_count": len(sources),
        }
