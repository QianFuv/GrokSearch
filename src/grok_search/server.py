"""
FastMCP server exposing Grok search and Tavily-backed web tools.
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from fastmcp import Context, FastMCP
from pydantic import Field

src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from grok_search.config import config
    from grok_search.logger import log_info
    from grok_search.providers.grok import GrokSearchProvider
    from grok_search.sources import (
        SourcesCache,
        merge_sources,
        new_session_id,
        split_answer_and_sources,
    )
except ImportError:
    from .config import config
    from .logger import log_info
    from .providers.grok import GrokSearchProvider
    from .sources import (
        SourcesCache,
        merge_sources,
        new_session_id,
        split_answer_and_sources,
    )

mcp = FastMCP("grok-search")

_SOURCES_CACHE = SourcesCache(max_size=256)
_AVAILABLE_MODELS_CACHE: dict[tuple[str, str], list[str]] = {}
_AVAILABLE_MODELS_LOCK = asyncio.Lock()


def _normalize_site_host(url: str) -> str | None:
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


def _hosts_match(base_host: str, candidate_host: str) -> bool:
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


def _extract_result_url(item: Any) -> str | None:
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


def _filter_same_site_results(base_url: str, results: list[Any]) -> list[Any]:
    """
    Keep only same-site map results for the requested base URL.

    Args:
        base_url: The original URL passed to the map tool.
        results: Raw result items returned by the upstream API.

    Returns:
        Result items whose hosts match the requested site.
    """
    base_host = _normalize_site_host(base_url)
    if not base_host:
        return results

    filtered: list[Any] = []
    for item in results:
        result_url = _extract_result_url(item)
        if not result_url:
            continue
        candidate_host = _normalize_site_host(result_url)
        if candidate_host and _hosts_match(base_host, candidate_host):
            filtered.append(item)
    return filtered


def _build_tavily_map_body(
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

    base_host = _normalize_site_host(url)
    if base_host:
        body["allow_external"] = False
        body["select_domains"] = [base_host]

    return body


async def _search_with_answer_retry(
    grok_provider: "GrokSearchProvider",
    query: str,
    platform: str,
    max_attempts: int = 2,
) -> str:
    """
    Retry a Grok search when the upstream response has no usable answer body.

    Args:
        grok_provider: The configured Grok provider instance.
        query: The search query text.
        platform: Optional platform focus hint.
        max_attempts: The maximum number of semantic retries.

    Returns:
        The final raw model response text.
    """
    final_result = ""
    total_attempts = max(1, max_attempts)

    for attempt in range(1, total_attempts + 1):
        final_result = await grok_provider.search(query, platform)
        answer, _ = split_answer_and_sources(final_result)
        if answer.strip():
            return final_result
        if attempt < total_attempts:
            await _sleep_before_retry(attempt)

    return final_result


async def _fetch_available_models(api_url: str, api_key: str) -> list[str]:
    """
    Fetch the available model identifiers from the upstream API.

    Args:
        api_url: The upstream API base URL.
        api_key: The API key used for authentication.

    Returns:
        A list of available model identifiers.
    """
    import httpx

    models_url = f"{api_url.rstrip('/')}/models"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            models_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

    models: list[str] = []
    for item in (data or {}).get("data", []) or []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            models.append(item["id"])
    return models


async def _get_available_models_cached(api_url: str, api_key: str) -> list[str]:
    """
    Fetch available models with an in-memory cache.

    Args:
        api_url: The upstream API base URL.
        api_key: The API key used for authentication.

    Returns:
        A cached or freshly fetched list of model identifiers.
    """
    key = (api_url, api_key)
    async with _AVAILABLE_MODELS_LOCK:
        if key in _AVAILABLE_MODELS_CACHE:
            return _AVAILABLE_MODELS_CACHE[key]

    try:
        models = await _fetch_available_models(api_url, api_key)
    except Exception:
        models = []

    async with _AVAILABLE_MODELS_LOCK:
        _AVAILABLE_MODELS_CACHE[key] = models
    return models


def _apply_model_suffix(api_url: str, model: str) -> str:
    """
    Normalize a model name for proxy-specific routing rules.

    Args:
        api_url: The upstream API base URL.
        model: The requested model identifier.

    Returns:
        The normalized model identifier.
    """
    if "openrouter" in api_url and ":online" not in model:
        return f"{model}:online"
    return model


def _is_available_model(
    api_url: str, requested_model: str, available_models: list[str]
) -> bool:
    """
    Check whether a requested model appears in the fetched model list.

    Args:
        api_url: The upstream API base URL.
        requested_model: The model requested by the caller.
        available_models: The fetched model identifiers.

    Returns:
        True when the requested model is available.
    """
    return any(
        candidate in available_models
        for candidate in (
            requested_model,
            _apply_model_suffix(api_url, requested_model),
        )
    )


async def _resolve_request_model(
    api_url: str, api_key: str, requested_model: str
) -> str:
    """
    Resolve the effective model for a request.

    Args:
        api_url: The upstream API base URL.
        api_key: The API key used for authentication.
        requested_model: The model explicitly requested by the caller.

    Returns:
        The effective model identifier for the request.

    Raises:
        ValueError: Raised when an explicit model is invalid.
    """
    if not requested_model:
        return config.grok_model

    available = await _get_available_models_cached(api_url, api_key)
    if available and not _is_available_model(api_url, requested_model, available):
        raise ValueError(f"Invalid model: {requested_model}")
    return _apply_model_suffix(api_url, requested_model)


async def _validate_model_selection(api_url: str, api_key: str, model: str) -> None:
    """
    Validate that a model exists in the upstream /models response.

    Args:
        api_url: The upstream API base URL.
        api_key: The API key used for authentication.
        model: The model identifier to validate.

    Returns:
        None.

    Raises:
        ValueError: Raised when validation fails or no models are returned.
    """
    try:
        available = await _fetch_available_models(api_url, api_key)
    except Exception as exc:
        raise ValueError(
            f"Unable to validate the model against /models: {type(exc).__name__}: {exc}"
        ) from exc

    if not available:
        raise ValueError(
            "Unable to validate the model because /models returned no data"
        )

    if not _is_available_model(api_url, model, available):
        preview = ", ".join(available[:10])
        suffix = ", ..." if len(available) > 10 else ""
        raise ValueError(f"Invalid model: {model}. Available models: {preview}{suffix}")


def _retry_delay_seconds(retry_index: int) -> float:
    """
    Compute the retry delay for a retry attempt index.

    Args:
        retry_index: The 1-based retry attempt index.

    Returns:
        The retry delay in seconds.
    """
    delay = config.retry_multiplier * (2 ** max(retry_index - 1, 0))
    return min(float(config.retry_max_wait), max(0.0, delay))


async def _sleep_before_retry(retry_index: int) -> None:
    """
    Sleep for the configured retry delay when positive.

    Args:
        retry_index: The 1-based retry attempt index.

    Returns:
        None.
    """
    delay = _retry_delay_seconds(retry_index)
    if delay > 0:
        await asyncio.sleep(delay)


def _extra_results_to_sources(tavily_results: list[dict] | None) -> list[dict]:
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
        for r in tavily_results:
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            source_item: dict = {"url": url, "provider": "tavily"}
            title = (r.get("title") or "").strip()
            if title:
                source_item["title"] = title
            content = (r.get("content") or "").strip()
            if content:
                source_item["description"] = content
            sources.append(source_item)

    return sources


@mcp.tool(
    name="web_search",
    output_schema=None,
    description="""
    Performs a deep web search based on the given query
    and returns Grok's answer directly.

    This tool extracts sources if provided by upstream, caches them, and returns:
    - session_id: string
      (Use this field with get_sources to inspect cached sources.)
    - content: string (answer only)
    - sources_count: int
    """,
    meta={"version": "2.0.0", "author": "grok-search"},
)
async def web_search(
    query: Annotated[str, "Clear, self-contained natural-language search query."],
    platform: Annotated[
        str,
        (
            "Target platform to focus on (for example, 'Twitter', 'GitHub', "
            "or 'Reddit'). Leave empty for general web search."
        ),
    ] = "",
    model: Annotated[
        str,
        (
            "Optional model ID for this request only. This value is used only "
            "when the user explicitly provided it."
        ),
    ] = "",
    extra_sources: Annotated[
        int,
        ("Number of additional reference results from Tavily. Set 0 to disable."),
    ] = 0,
) -> dict:
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
        api_url = config.grok_api_url
        api_key = config.grok_api_key
    except ValueError as e:
        await _SOURCES_CACHE.set(session_id, [])
        return {
            "session_id": session_id,
            "content": f"Configuration error: {e}",
            "sources_count": 0,
        }

    try:
        effective_model = await _resolve_request_model(api_url, api_key, model)
    except ValueError as e:
        await _SOURCES_CACHE.set(session_id, [])
        return {
            "session_id": session_id,
            "content": str(e),
            "sources_count": 0,
        }

    grok_provider = GrokSearchProvider(api_url, api_key, effective_model)

    tavily_count = extra_sources if extra_sources > 0 and config.tavily_api_key else 0

    async def _safe_grok() -> str:
        """
        Execute the Grok search path with semantic answer retries.

        Returns:
            The raw Grok response text.
        """
        return await _search_with_answer_retry(grok_provider, query, platform)

    async def _safe_tavily() -> list[dict] | None:
        """
        Fetch extra Tavily sources while swallowing optional failures.

        Returns:
            A list of Tavily results, or None when unavailable.
        """
        try:
            if tavily_count:
                return await _call_tavily_search(query, tavily_count)
        except Exception:
            return None
        return None

    coros: list = [_safe_grok()]
    if tavily_count > 0:
        coros.append(_safe_tavily())

    gathered = list(await asyncio.gather(*coros, return_exceptions=True))

    grok_first = gathered[0]
    grok_error: Exception | None = (
        grok_first if isinstance(grok_first, Exception) else None
    )
    grok_result = grok_first if isinstance(grok_first, str) else ""
    tavily_results: list[dict] | None = None
    if tavily_count > 0:
        tavily_value = gathered[1]
        tavily_results = tavily_value if isinstance(tavily_value, list) else None

    answer, grok_sources = split_answer_and_sources(grok_result)
    extra = _extra_results_to_sources(tavily_results)
    all_sources = merge_sources(grok_sources, extra)

    if not answer:
        if grok_error is not None:
            answer = f"Search upstream error: {type(grok_error).__name__}: {grok_error}"
        elif all_sources:
            answer = (
                "Search completed, but the upstream response did not contain a "
                "parsable answer body. Call get_sources to inspect the sources."
            )
        else:
            answer = (
                "Search completed, but the upstream response was empty after retries."
            )

    await _SOURCES_CACHE.set(session_id, all_sources)
    return {
        "session_id": session_id,
        "content": answer,
        "sources_count": len(all_sources),
    }


@mcp.tool(
    name="get_sources",
    description="""
    Use the session_id returned by web_search to obtain the corresponding
    cached source list.
    Retrieve all cached sources for a previous web_search call.
    Provide the session_id returned by web_search to get the full source list.
    """,
    meta={"version": "1.0.0", "author": "grok-search"},
)
async def get_sources(
    session_id: Annotated[str, "Session ID from previous web_search call."],
) -> dict:
    """
    Retrieve cached sources for a previous web_search session.

    Args:
        session_id: The session identifier returned by web_search.

    Returns:
        A dictionary containing the cached sources or an expiration error.
    """
    sources = await _SOURCES_CACHE.get(session_id)
    if sources is None:
        return {
            "session_id": session_id,
            "sources": [],
            "sources_count": 0,
            "error": "session_id_not_found_or_expired",
        }
    return {"session_id": session_id, "sources": sources, "sources_count": len(sources)}


async def _call_tavily_extract(url: str) -> str | None:
    """
    Call Tavily extract until structured content becomes available.

    Args:
        url: The page URL to extract.

    Returns:
        The extracted Markdown content, or None when unavailable.
    """
    import httpx

    api_url = config.tavily_api_url
    api_key = config.tavily_api_key
    if not api_key:
        return None
    endpoint = f"{api_url.rstrip('/')}/extract"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"urls": [url], "format": "markdown"}
    max_attempts = max(1, config.retry_max_attempts)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(1, max_attempts + 1):
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

            if attempt < max_attempts:
                await _sleep_before_retry(attempt)

    return None


async def _call_tavily_search(query: str, max_results: int = 6) -> list[dict] | None:
    """
    Call Tavily search and normalize its search results.

    Args:
        query: The search query text.
        max_results: The maximum number of search results to request.

    Returns:
        A normalized result list, or None when unavailable.
    """
    import httpx

    api_key = config.tavily_api_key
    if not api_key:
        return None
    endpoint = f"{config.tavily_api_url.rstrip('/')}/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_raw_content": False,
        "include_answer": False,
    }
    max_attempts = max(1, config.retry_max_attempts)

    async with httpx.AsyncClient(timeout=90.0) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.post(endpoint, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                return (
                    [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "content": r.get("content", ""),
                            "score": r.get("score", 0),
                        }
                        for r in results
                    ]
                    if results
                    else None
                )
            except Exception:
                if attempt >= max_attempts:
                    break
                await _sleep_before_retry(attempt)
    return None


@mcp.tool(
    name="web_fetch",
    output_schema=None,
    description="""
    Fetches and extracts complete content from a URL,
    returning it as a structured Markdown document.

    **Key Features:**
        - **Full Content Extraction:** Retrieves and parses meaningful
          content such as text, images, links, tables, and code blocks.
        - **Markdown Conversion:** Converts HTML structure to
          well-formatted Markdown with preserved hierarchy.
        - **Content Fidelity:** Maintains full content fidelity
          without summarization or modification.

    **Edge Cases & Best Practices:**
        - Ensure URL is complete and accessible (not behind authentication or paywalls).
        - May not capture dynamically loaded content requiring JavaScript execution.
        - Large pages may take longer to process; consider timeout implications.
    """,
    meta={"version": "1.3.0", "author": "grok-search"},
)
async def web_fetch(
    url: Annotated[
        str,
        (
            "Valid HTTP or HTTPS web address pointing to the target page. "
            "It must be complete and accessible."
        ),
    ],
    ctx: Context | None = None,
) -> str:
    """
    Fetch and extract a webpage into Markdown using Tavily.

    Args:
        url: The target page URL.
        ctx: Optional FastMCP context used for logging.

    Returns:
        Extracted Markdown content or an error message.
    """
    await log_info(ctx, f"Begin Fetch: {url}", config.debug_enabled)

    if not config.tavily_api_key:
        return "Configuration error: TAVILY_API_KEY is not configured"

    result = await _call_tavily_extract(url)
    if result:
        await log_info(ctx, "Fetch Finished (Tavily)!", config.debug_enabled)
        return result

    await log_info(ctx, "Fetch Failed!", config.debug_enabled)
    return "Extraction failed: Tavily did not return content after retries"


async def _call_tavily_map(
    url: str,
    instructions: str | None = None,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    timeout: int = 150,
) -> str:
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
        A JSON string describing the map response or an error message.
    """
    import json

    import httpx

    async def _post_map_request(
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

    api_url = config.tavily_api_url
    api_key = config.tavily_api_key
    if not api_key:
        return "Configuration error: TAVILY_API_KEY is not configured"
    endpoint = f"{api_url.rstrip('/')}/map"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = _build_tavily_map_body(
        url=url,
        instructions=instructions,
        max_depth=max_depth,
        max_breadth=max_breadth,
        limit=limit,
        timeout=timeout,
    )
    try:
        async with httpx.AsyncClient(timeout=float(timeout + 10)) as client:
            try:
                data = await _post_map_request(client, body)
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
                data = await _post_map_request(client, legacy_body)
            raw_results = data.get("results", [])
            results = (
                _filter_same_site_results(url, raw_results)
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
                    legacy_data = await _post_map_request(client, legacy_body)
                except httpx.HTTPStatusError:
                    legacy_data = {}
                legacy_results = legacy_data.get("results", [])
                legacy_filtered = (
                    _filter_same_site_results(url, legacy_results)
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
            return json.dumps(
                response_payload,
                ensure_ascii=False,
                indent=2,
            )
    except httpx.TimeoutException:
        return f"Map timeout: the request exceeded {timeout} seconds"
    except httpx.HTTPStatusError as e:
        return f"HTTP error: {e.response.status_code} - {e.response.text[:200]}"
    except Exception as e:
        return f"Map error: {e}"


@mcp.tool(
    name="web_map",
    description="""
    Maps a website's structure by traversing it like a graph,
    discovering URLs and generating a comprehensive site map.

    **Key Features:**
        - **Graph Traversal:** Explores website structure starting from root URL.
        - **Depth & Breadth Control:** Configures traversal limits to balance
          coverage and performance.
        - **Instruction Filtering:** Uses natural language to focus the
          crawler on specific content types.

    **Edge Cases & Best Practices:**
        - Start with low max_depth (1-2) for initial exploration.
        - Use instructions to filter for specific content.
        - Large sites may hit timeout limits, so adjust timeout and limit
          parameters accordingly.
    """,
    meta={"version": "1.3.0", "author": "grok-search"},
)
async def web_map(
    url: Annotated[
        str, "Root URL to begin the mapping (e.g., 'https://docs.example.com')."
    ],
    instructions: Annotated[
        str,
        (
            "Natural language instructions for the crawler to filter or focus "
            "on specific content."
        ),
    ] = "",
    max_depth: Annotated[
        int,
        Field(description="Maximum depth of mapping from the base URL.", ge=1, le=5),
    ] = 1,
    max_breadth: Annotated[
        int,
        Field(description="Maximum number of links to follow per page.", ge=1, le=500),
    ] = 20,
    limit: Annotated[
        int,
        Field(
            description="Total number of links to process before stopping.",
            ge=1,
            le=500,
        ),
    ] = 50,
    timeout: Annotated[
        int,
        Field(description="Maximum time in seconds for the operation.", ge=10, le=150),
    ] = 150,
) -> str:
    """
    Traverse a website and return a same-site map response.

    Args:
        url: The root URL to map.
        instructions: Optional natural-language crawl instructions.
        max_depth: The maximum crawl depth.
        max_breadth: The maximum breadth per page.
        limit: The total result limit.
        timeout: The request timeout in seconds.

    Returns:
        A JSON string describing the map response.
    """
    result = await _call_tavily_map(
        url, instructions, max_depth, max_breadth, limit, timeout
    )
    return result


@mcp.tool(
    name="get_config_info",
    output_schema=None,
    description="""
    Returns current Grok Search MCP server configuration and tests API connectivity.

    **Key Features:**
        - **Configuration Check:** Verifies environment variables and current settings.
        - **Connection Test:** Sends request to /models endpoint to validate API access.
        - **Model Discovery:** Lists all available models from the API.

    **Edge Cases & Best Practices:**
        - Use this tool first when debugging connection or configuration issues.
        - API keys are automatically masked for security in the response.
        - Connection test timeout is 10 seconds; network issues may cause delays.
    """,
    meta={"version": "1.3.0", "author": "grok-search"},
)
async def get_config_info() -> str:
    """
    Return current configuration details and connectivity diagnostics.

    Returns:
        A JSON string containing configuration details and connection status.
    """
    import json
    import time

    import httpx

    config_info = config.get_config_info()

    test_result: dict[str, object] = {
        "status": "not_tested",
        "message": "",
        "response_time_ms": 0,
    }

    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key

        start_time = time.time()
        model_names = await _fetch_available_models(api_url, api_key)
        response_time = (time.time() - start_time) * 1000

        test_result["status"] = "connected"
        test_result["response_time_ms"] = round(response_time, 2)
        if model_names:
            test_result["message"] = (
                f"Fetched model list successfully; {len(model_names)} models available"
            )
            test_result["available_models"] = model_names
        else:
            test_result["message"] = (
                "Fetched model list successfully, but the API returned no models"
            )

    except httpx.TimeoutException:
        test_result["status"] = "timeout"
        test_result["message"] = (
            "The request timed out after 10 seconds. "
            "Check network access or the API URL."
        )
    except httpx.RequestError as e:
        test_result["status"] = "request_error"
        test_result["message"] = f"Network error: {e}"
    except ValueError as e:
        test_result["status"] = "configuration_error"
        test_result["message"] = str(e)
    except Exception as e:
        test_result["status"] = "failed"
        test_result["message"] = f"Unexpected error: {e}"

    config_info["connection_test"] = test_result

    return json.dumps(config_info, ensure_ascii=False, indent=2)


@mcp.tool(
    name="switch_model",
    output_schema=None,
    description="""
    Switches the default Grok model used for search and fetch operations,
    persisting the setting.

    **Key Features:**
        - **Model Selection:** Change the AI model for web search and content fetching.
        - **Persistent Storage:** Model preference saved to
          ~/.config/grok-search/config.json.
        - **Immediate Effect:** New model used for all subsequent operations.

    **Edge Cases & Best Practices:**
        - Use get_config_info to verify available models before switching.
        - Invalid model IDs may cause API errors in subsequent requests.
        - Model changes persist across sessions until explicitly changed again.
    """,
    meta={"version": "1.3.0", "author": "grok-search"},
)
async def switch_model(
    model: Annotated[
        str,
        (
            "Model ID to switch to (for example, 'grok-4-fast', "
            "'grok-2-latest', or 'grok-vision-beta')."
        ),
    ],
) -> str:
    """
    Persist a new default Grok model after validating it.

    Args:
        model: The model identifier to persist.

    Returns:
        A JSON string describing the update outcome.
    """
    import json

    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key
        previous_model = config.grok_model
        await _validate_model_selection(api_url, api_key, model)
        config.set_model(model)
        current_model = config.grok_model

        result = {
            "status": "success",
            "previous_model": previous_model,
            "current_model": current_model,
            "message": f"Switched the model from {previous_model} to {current_model}",
            "config_file": str(config.config_file),
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except ValueError as e:
        result = {"status": "failed", "message": f"Failed to switch models: {e}"}
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        result = {"status": "failed", "message": f"Unexpected error: {e}"}
        return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool(
    name="describe_url",
    output_schema=None,
    description="""
    Ask Grok to read a single page and return a title plus verbatim extracts.
    """,
    meta={"version": "1.3.0", "author": "grok-search"},
)
async def describe_url(
    url: Annotated[
        str,
        "Valid HTTP/HTTPS web address pointing to the target page.",
    ],
    model: Annotated[
        str,
        (
            "Optional model ID for this request only. This value is used only "
            "when the user explicitly provided it."
        ),
    ] = "",
    ctx: Context | None = None,
) -> dict:
    """
    Ask Grok to describe a single page and extract verbatim snippets.

    Args:
        url: The target page URL.
        model: An optional per-request model override.
        ctx: Optional FastMCP context used for logging.

    Returns:
        A dictionary containing the page description result.
    """
    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key
        effective_model = await _resolve_request_model(api_url, api_key, model)
    except ValueError as e:
        return {"url": url, "title": url, "extracts": "", "error": str(e)}

    provider = GrokSearchProvider(api_url, api_key, effective_model)
    try:
        return await provider.describe_url(url, ctx)
    except Exception as e:
        return {
            "url": url,
            "title": url,
            "extracts": "",
            "error": f"Describe URL failed: {type(e).__name__}: {e}",
        }


@mcp.tool(
    name="rank_sources",
    output_schema=None,
    description="""
    Ask Grok to reorder a numbered source list by relevance to a query.
    """,
    meta={"version": "1.3.0", "author": "grok-search"},
)
async def rank_sources(
    query: Annotated[str, "The user query to rank sources against."],
    sources_text: Annotated[
        str,
        (
            "A numbered source list in plain text or Markdown. "
            "Every source number must appear exactly once."
        ),
    ],
    total: Annotated[
        int,
        Field(description="The number of sources in the list.", ge=1),
    ],
    model: Annotated[
        str,
        (
            "Optional model ID for this request only. This value is used only "
            "when the user explicitly provided it."
        ),
    ] = "",
    ctx: Context | None = None,
) -> dict:
    """
    Ask Grok to rank a numbered source list against a query.

    Args:
        query: The user query.
        sources_text: The numbered source list.
        total: The total number of sources in the list.
        model: An optional per-request model override.
        ctx: Optional FastMCP context used for logging.

    Returns:
        A dictionary containing the ranked source order or an error.
    """
    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key
        effective_model = await _resolve_request_model(api_url, api_key, model)
    except ValueError as e:
        return {"query": query, "order": [], "error": str(e)}

    provider = GrokSearchProvider(api_url, api_key, effective_model)
    try:
        order = await provider.rank_sources(query, sources_text, total, ctx)
    except Exception as e:
        return {
            "query": query,
            "order": [],
            "error": f"Rank sources failed: {type(e).__name__}: {e}",
        }
    return {"query": query, "order": order, "total": total}


def main():
    """
    Run the FastMCP server and ensure child shutdown on parent exit.

    Returns:
        None.
    """
    import os
    import signal
    import threading

    if threading.current_thread() is threading.main_thread():

        def handle_shutdown(signum, frame):
            """
            Exit immediately when a shutdown signal is received.

            Args:
                signum: The received signal number.
                frame: The current stack frame.

            Returns:
                None.
            """
            os._exit(0)

        signal.signal(signal.SIGINT, handle_shutdown)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, handle_shutdown)

    if sys.platform == "win32":
        import ctypes
        import time

        parent_pid = os.getppid()

        def is_parent_alive(pid):
            """
            Check whether the parent process is still alive on Windows.

            Args:
                pid: The parent process identifier.

            Returns:
                True when the parent process is still running.
            """
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return True
            exit_code = ctypes.c_ulong()
            result = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return result and exit_code.value == STILL_ACTIVE

        def monitor_parent():
            """
            Exit the process when the parent process disappears.

            Returns:
                None.
            """
            while True:
                if not is_parent_alive(parent_pid):
                    os._exit(0)
                time.sleep(2)

        threading.Thread(target=monitor_parent, daemon=True).start()

    try:
        mcp.run(transport="stdio", show_banner=False)
    except KeyboardInterrupt:
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
