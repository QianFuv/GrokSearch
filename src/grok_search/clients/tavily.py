"""Tavily client helpers for extract, search, and map operations."""

from typing import Any
from urllib.parse import urlparse

import httpx

from ..config import Config
from ..runtime import get_retry_settings, sleep_before_retry


def normalize_site_host(url: str) -> str | None:
    """
    Normalize a URL into a lowercase hostname for same-site filtering.

    Args:
        url: The URL to inspect.

    Returns:
        The normalized hostname, or None when it cannot be determined.
    """
    parsed = urlparse(url)
    if not parsed.hostname:
        return None
    return parsed.hostname.lower().rstrip(".")


def hosts_match(base_host: str, candidate_host: str) -> bool:
    """
    Check whether two hosts should be treated as the same site.

    Args:
        base_host: The hostname from the requested base URL.
        candidate_host: The hostname from a discovered result URL.

    Returns:
        True when both hosts belong to the same site.
    """
    normalized_base = base_host.removeprefix("www.")
    normalized_candidate = candidate_host.removeprefix("www.")
    return normalized_base == normalized_candidate


def extract_result_url(item: Any) -> str | None:
    """
    Extract a URL string from a Tavily map result item.

    Args:
        item: A raw result item from the upstream map API.

    Returns:
        The discovered URL, or None when no usable URL exists.
    """
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        for key in ("url", "href", "link"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value

    return None


def filter_same_site_results(base_url: str, results: list[Any]) -> list[Any]:
    """
    Keep only same-site map results for the requested base URL.

    Args:
        base_url: The original URL passed to the map tool.
        results: Raw result items returned by the upstream API.

    Returns:
        Result items whose hosts match the requested site.
    """
    base_host = normalize_site_host(base_url)
    if not base_host:
        return results

    filtered: list[Any] = []
    for item in results:
        result_url = extract_result_url(item)
        if not result_url:
            continue
        candidate_host = normalize_site_host(result_url)
        if candidate_host and hosts_match(base_host, candidate_host):
            filtered.append(item)
    return filtered


def build_tavily_map_body(
    url: str,
    instructions: str | None,
    max_depth: int,
    max_breadth: int,
    limit: int,
    timeout: int,
) -> dict[str, Any]:
    """
    Build a Tavily-compatible map request with same-site defaults.

    Args:
        url: The starting URL for the map request.
        instructions: Optional natural-language filtering instructions.
        max_depth: The maximum crawl depth.
        max_breadth: The maximum breadth per page.
        limit: The total result limit.
        timeout: The request timeout in seconds.

    Returns:
        The request payload sent to the upstream map API.
    """
    body: dict[str, Any] = {
        "url": url,
        "max_depth": max_depth,
        "max_breadth": max_breadth,
        "limit": limit,
        "timeout": timeout,
    }
    if instructions:
        body["instructions"] = instructions

    base_host = normalize_site_host(url)
    if base_host:
        body["allow_external"] = False
        body["select_domains"] = [base_host]

    return body


class TavilyClient:
    """
    Call Tavily extract, search, and map endpoints.

    Attributes:
        _config: The runtime configuration instance.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the Tavily client.

        Args:
            config: The runtime configuration instance.

        Returns:
            None.
        """
        self._config = config

    @property
    def is_configured(self) -> bool:
        """
        Check whether Tavily credentials are configured.

        Returns:
            True when a Tavily API key is available.
        """
        return bool(self._config.tavily_api_key)

    def _get_headers(self) -> dict[str, str]:
        """
        Build the Tavily request headers.

        Returns:
            The Tavily request headers.

        Raises:
            ValueError: Raised when the Tavily API key is not configured.
        """
        api_key = self._config.tavily_api_key
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not configured")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def extract(self, url: str) -> str | None:
        """
        Call Tavily extract until structured content becomes available.

        Args:
            url: The page URL to extract.

        Returns:
            The extracted Markdown content, or None when unavailable.
        """
        if not self.is_configured:
            return None

        endpoint = f"{self._config.tavily_api_url.rstrip('/')}/extract"
        headers = self._get_headers()
        body = {"urls": [url], "format": "markdown"}
        retry_settings = get_retry_settings(self._config)

        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(1, retry_settings.max_attempts + 1):
                try:
                    response = await client.post(endpoint, headers=headers, json=body)
                    response.raise_for_status()
                    data = response.json()
                    results = data.get("results") or []
                    if results:
                        content = results[0].get("raw_content", "")
                        if content and content.strip():
                            return content
                except Exception:
                    pass

                if attempt < retry_settings.max_attempts:
                    await sleep_before_retry(attempt, retry_settings)

        return None

    async def search(self, query: str, max_results: int = 6) -> list[dict] | None:
        """
        Call Tavily search and normalize its search results.

        Args:
            query: The search query text.
            max_results: The maximum number of search results to request.

        Returns:
            A normalized result list, or None when unavailable.
        """
        if not self.is_configured:
            return None

        endpoint = f"{self._config.tavily_api_url.rstrip('/')}/search"
        headers = self._get_headers()
        body = {
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_raw_content": False,
            "include_answer": False,
        }
        retry_settings = get_retry_settings(self._config)

        async with httpx.AsyncClient(timeout=90.0) as client:
            for attempt in range(1, retry_settings.max_attempts + 1):
                try:
                    response = await client.post(endpoint, headers=headers, json=body)
                    response.raise_for_status()
                    data = response.json()
                    results = data.get("results", [])
                    return (
                        [
                            {
                                "title": result.get("title", ""),
                                "url": result.get("url", ""),
                                "content": result.get("content", ""),
                                "score": result.get("score", 0),
                            }
                            for result in results
                        ]
                        if results
                        else None
                    )
                except Exception:
                    if attempt >= retry_settings.max_attempts:
                        break
                    await sleep_before_retry(attempt, retry_settings)
        return None

    async def map_site(
        self,
        url: str,
        instructions: str | None = None,
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 50,
        timeout: int = 150,
    ) -> dict[str, Any]:
        """
        Call Tavily map with same-site defaults and fallback behavior.

        Args:
            url: The root URL to map.
            instructions: Optional natural-language crawl instructions.
            max_depth: The maximum crawl depth.
            max_breadth: The maximum breadth per page.
            limit: The total result limit.
            timeout: The request timeout in seconds.

        Returns:
            A dictionary describing the map response.

        Raises:
            ValueError: Raised when Tavily is not configured.
            httpx.TimeoutException: Raised when the upstream request times out.
            httpx.HTTPStatusError: Raised when the upstream returns an HTTP error.
        """

        async def post_map_request(
            client: httpx.AsyncClient, request_body: dict[str, Any]
        ) -> dict[str, Any]:
            """
            Post a map request and return the decoded JSON payload.

            Args:
                client: The shared async HTTP client.
                request_body: The JSON request body for the upstream map API.

            Returns:
                The decoded JSON response body.
            """
            response = await client.post(endpoint, headers=headers, json=request_body)
            response.raise_for_status()
            return response.json()

        endpoint = f"{self._config.tavily_api_url.rstrip('/')}/map"
        headers = self._get_headers()
        body = build_tavily_map_body(
            url=url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            timeout=timeout,
        )

        async with httpx.AsyncClient(timeout=float(timeout + 10)) as client:
            try:
                data = await post_map_request(client, body)
            except httpx.HTTPStatusError as exc:
                if (
                    exc.response.status_code != 400
                    or "allow_external" not in exc.response.text
                    and "select_domains" not in exc.response.text
                ):
                    raise
                legacy_body = {
                    key: value
                    for key, value in body.items()
                    if key not in {"allow_external", "select_domains"}
                }
                data = await post_map_request(client, legacy_body)

            raw_results = data.get("results", [])
            results = (
                filter_same_site_results(url, raw_results)
                if isinstance(raw_results, list)
                else []
            )

            response_payload: dict[str, Any] = {
                "base_url": data.get("base_url", ""),
                "results": results,
                "response_time": data.get("response_time", 0),
            }

            if isinstance(raw_results, list) and raw_results and not results:
                response_payload["raw_count"] = len(raw_results)
                response_payload["filtered_out_count"] = len(raw_results)
            elif (
                isinstance(raw_results, list)
                and not raw_results
                and "allow_external" in body
                and "select_domains" in body
            ):
                legacy_body = {
                    key: value
                    for key, value in body.items()
                    if key not in {"allow_external", "select_domains"}
                }
                try:
                    legacy_data = await post_map_request(client, legacy_body)
                except httpx.HTTPStatusError:
                    legacy_data = {}
                legacy_results = legacy_data.get("results", [])
                legacy_filtered = (
                    filter_same_site_results(url, legacy_results)
                    if isinstance(legacy_results, list)
                    else []
                )
                if (
                    isinstance(legacy_results, list)
                    and legacy_results
                    and not legacy_filtered
                ):
                    response_payload["raw_count"] = len(legacy_results)
                    response_payload["filtered_out_count"] = len(legacy_results)

            return response_payload
