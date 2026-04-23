"""
Grok provider implementations for search and page-level tools.
"""

import json
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.wait import wait_base

from ..logger import log_info
from ..prompts import RANK_SOURCES_PROMPT, SEARCH_PROMPT, URL_DESCRIBE_PROMPT
from ..runtime import RetrySettings
from ..utils import sanitize_model_output
from .base import BaseSearchProvider


def _extract_text_segments(content: Any) -> list[str]:
    """
    Normalize OpenAI-compatible content payloads into plain text segments.

    Args:
        content: A content payload from an SSE delta or message object.

    Returns:
        A list of extracted text fragments in the original order.
    """
    if isinstance(content, str):
        return [content]

    if isinstance(content, list):
        segments: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    segments.append(text)
                    continue
                nested = item.get("content")
                if nested is not None:
                    segments.extend(_extract_text_segments(nested))
            elif isinstance(item, str) and item:
                segments.append(item)
        return segments

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str) and text:
            return [text]
        nested = content.get("content")
        if nested is not None:
            return _extract_text_segments(nested)

    return []


def _extract_choice_text(choice: dict[str, Any]) -> str:
    """
    Extract text content from a single chat completion choice payload.

    Args:
        choice: A single completion choice object.

    Returns:
        The extracted text for that choice, or an empty string.
    """
    text_segments: list[str] = []

    delta = choice.get("delta")
    if isinstance(delta, dict):
        for key in ("content", "text"):
            text_segments.extend(_extract_text_segments(delta.get(key)))

    message = choice.get("message")
    if isinstance(message, dict):
        for key in ("content", "text"):
            text_segments.extend(_extract_text_segments(message.get(key)))

    for key in ("content", "text"):
        text_segments.extend(_extract_text_segments(choice.get(key)))

    return "".join(segment for segment in text_segments if segment)


def _extract_text_from_payload(payload: dict[str, Any]) -> str:
    """
    Extract response text from a streamed or non-streamed completion payload.

    Args:
        payload: A decoded JSON payload from the upstream API.

    Returns:
        The extracted text fragment, or an empty string.
    """
    choices = payload.get("choices")
    if isinstance(choices, list):
        fragments = [
            _extract_choice_text(choice)
            for choice in choices
            if isinstance(choice, dict)
        ]
        return "".join(fragment for fragment in fragments if fragment)

    return ""


def get_local_time_info() -> str:
    """
    Build local time context for time-sensitive search queries.

    Returns:
        A formatted time context block for prompt injection.
    """
    try:
        local_tz = datetime.now().astimezone().tzinfo
        local_now = datetime.now(local_tz)
    except Exception:
        local_now = datetime.now(UTC)

    weekdays_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays_cn[local_now.weekday()]

    return (
        f"[Current Time Context]\n"
        f"- Date: {local_now.strftime('%Y-%m-%d')} ({weekday})\n"
        f"- Time: {local_now.strftime('%H:%M:%S')}\n"
        f"- Timezone: {local_now.tzname() or 'Local'}\n"
    )


def _needs_time_context(query: str) -> bool:
    """
    Check whether a query likely depends on current time context.

    Args:
        query: The user search query.

    Returns:
        True when time context should be injected into the prompt.
    """
    cn_keywords = [
        "当前",
        "现在",
        "今天",
        "明天",
        "昨天",
        "本周",
        "上周",
        "下周",
        "这周",
        "本月",
        "上月",
        "下月",
        "这个月",
        "今年",
        "去年",
        "明年",
        "最新",
        "最近",
        "近期",
        "刚刚",
        "刚才",
        "实时",
        "即时",
        "目前",
    ]
    en_keywords = [
        "current",
        "now",
        "today",
        "tomorrow",
        "yesterday",
        "this week",
        "last week",
        "next week",
        "this month",
        "last month",
        "next month",
        "this year",
        "last year",
        "next year",
        "latest",
        "recent",
        "recently",
        "just now",
        "real-time",
        "realtime",
        "up-to-date",
    ]

    query_lower = query.lower()

    for keyword in cn_keywords:
        if keyword in query:
            return True

    return any(keyword in query_lower for keyword in en_keywords)


RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _is_retryable_exception(exc) -> bool:
    """
    Check whether an upstream exception should trigger a retry.

    Args:
        exc: The exception raised by the HTTP client.

    Returns:
        True when the exception matches retry policy.
    """
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


class _WaitWithRetryAfter(wait_base):
    """
    Prefer Retry-After headers before falling back to exponential backoff.

    Attributes:
        _base_wait: The fallback exponential backoff strategy.
        _protocol_error_base: Extra delay applied after protocol errors.
    """

    def __init__(self, multiplier: float, max_wait: int):
        """
        Initialize the retry wait strategy.

        Args:
            multiplier: The exponential backoff multiplier.
            max_wait: The maximum wait duration in seconds.

        Returns:
            None.
        """
        self._base_wait = wait_random_exponential(multiplier=multiplier, max=max_wait)
        self._protocol_error_base = 3.0

    def __call__(self, retry_state):
        """
        Compute the next retry delay for the given retry state.

        Args:
            retry_state: The Tenacity retry state object.

        Returns:
            The delay in seconds before the next retry.
        """
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            if (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code == 429
            ):
                retry_after = self._parse_retry_after(exc.response)
                if retry_after is not None:
                    return retry_after
            if isinstance(exc, httpx.RemoteProtocolError):
                return self._base_wait(retry_state) + self._protocol_error_base
        return self._base_wait(retry_state)

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        """
        Parse a Retry-After header value from an HTTP response.

        Args:
            response: The HTTP response carrying the Retry-After header.

        Returns:
            The retry delay in seconds, or None when parsing fails.
        """
        header = response.headers.get("Retry-After")
        if not header:
            return None
        header = header.strip()

        if header.isdigit():
            return float(header)

        try:
            retry_dt = parsedate_to_datetime(header)
            if retry_dt.tzinfo is None:
                retry_dt = retry_dt.replace(tzinfo=UTC)
            delay = (retry_dt - datetime.now(UTC)).total_seconds()
            return max(0.0, delay)
        except (TypeError, ValueError):
            return None


class GrokSearchProvider(BaseSearchProvider):
    """
    Call Grok-compatible endpoints for search and page-level tools.

    Attributes:
        model: The model identifier used for requests.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "grok-4-fast",
        retry_settings: RetrySettings | None = None,
        debug_enabled: bool = False,
    ) -> None:
        """
        Initialize the Grok search provider.

        Args:
            api_url: The upstream API base URL.
            api_key: The API key used for authentication.
            model: The model identifier used for requests.
            retry_settings: Retry timing settings for upstream requests.
            debug_enabled: Whether debug logging is enabled.

        Returns:
            None.
        """
        super().__init__(api_url, api_key)
        self.model = model
        self.retry_settings = retry_settings or RetrySettings(
            max_attempts=3,
            multiplier=1.0,
            max_wait=10,
        )
        self.debug_enabled = debug_enabled

    def get_provider_name(self) -> str:
        """
        Return the provider display name.

        Returns:
            The provider name.
        """
        return "Grok"

    async def search(
        self,
        query: str,
        platform: str = "",
        min_results: int = 3,
        max_results: int = 10,
        ctx=None,
    ) -> str:
        """
        Execute a Grok-backed web search request.

        Args:
            query: The user search query.
            platform: An optional platform focus hint.
            min_results: The minimum desired result count.
            max_results: The maximum desired result count.
            ctx: Optional FastMCP context used for logging.

        Returns:
            The raw model response text.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        platform_prompt = ""

        if platform:
            platform_prompt = (
                "\n\nYou should search the web for the information you need, "
                "and focus on these platform: "
                f"{platform}\n"
            )

        time_context = (
            get_local_time_info() + "\n" if _needs_time_context(query) else ""
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SEARCH_PROMPT,
                },
                {"role": "user", "content": time_context + query + platform_prompt},
            ],
            "stream": True,
        }

        await log_info(
            ctx, f"platform_prompt: {query + platform_prompt}", self.debug_enabled
        )

        return await self._execute_stream_with_retry(headers, payload, ctx)

    async def _parse_streaming_response(
        self, response: httpx.Response, ctx=None
    ) -> str:
        """
        Parse SSE or JSON completion responses into cleaned text.

        Args:
            response: The upstream HTTP response object.
            ctx: Optional FastMCP context used for logging.

        Returns:
            The cleaned model output text.
        """
        content = ""
        full_body_buffer: list[str] = []

        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue

            full_body_buffer.append(line)

            if line.startswith("event:"):
                continue

            payload_text = line[5:].lstrip() if line.startswith("data:") else line
            if payload_text == "[DONE]":
                continue

            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue

            content += _extract_text_from_payload(payload)

        if not content and full_body_buffer:
            for item in full_body_buffer:
                if item.startswith("event:"):
                    continue
                payload_text = item[5:].lstrip() if item.startswith("data:") else item
                if payload_text == "[DONE]":
                    continue
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue
                content += _extract_text_from_payload(payload)

        content = sanitize_model_output(content)

        await log_info(ctx, f"content: {content}", self.debug_enabled)

        return content

    async def _execute_stream_with_retry(
        self, headers: dict[str, str], payload: dict[str, Any], ctx=None
    ) -> str:
        """
        Execute a streaming completion request with retry handling.

        Args:
            headers: The HTTP headers for the request.
            payload: The JSON payload sent to the upstream API.
            ctx: Optional FastMCP context used for logging.

        Returns:
            The cleaned model response text.
        """
        timeout = httpx.Timeout(connect=6.0, read=120.0, write=10.0, pool=None)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.retry_settings.max_attempts + 1),
                wait=_WaitWithRetryAfter(
                    self.retry_settings.multiplier,
                    self.retry_settings.max_wait,
                ),
                retry=retry_if_exception(_is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    async with client.stream(
                        "POST",
                        f"{self.api_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        return await self._parse_streaming_response(response, ctx)
        raise RuntimeError("Search request ended without a response")

    async def describe_url(self, url: str, ctx=None) -> dict:
        """
        Ask Grok to inspect a URL and return a title with extracts.

        Args:
            url: The target page URL.
            ctx: Optional FastMCP context used for logging.

        Returns:
            A dictionary containing the page title, extracts, and URL.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": URL_DESCRIBE_PROMPT},
                {"role": "user", "content": url},
            ],
            "stream": True,
        }
        result = await self._execute_stream_with_retry(headers, payload, ctx)
        title, extracts = url, ""
        for line in result.strip().splitlines():
            if line.startswith("Title:"):
                title = line[6:].strip() or url
            elif line.startswith("Extracts:"):
                extracts = line[9:].strip()
        return {"title": title, "extracts": extracts, "url": url}

    async def rank_sources(
        self, query: str, sources_text: str, total: int, ctx=None
    ) -> list[int]:
        """
        Ask Grok to rank numbered sources by relevance to a query.

        Args:
            query: The user query.
            sources_text: The numbered source list text.
            total: The total number of numbered sources.
            ctx: Optional FastMCP context used for logging.

        Returns:
            A list of ranked source numbers.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": RANK_SOURCES_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\n{sources_text}"},
            ],
            "stream": True,
        }
        result = await self._execute_stream_with_retry(headers, payload, ctx)
        order: list[int] = []
        seen: set[int] = set()
        for token in result.strip().split():
            try:
                n = int(token)
                if 1 <= n <= total and n not in seen:
                    seen.add(n)
                    order.append(n)
            except ValueError:
                continue
        for i in range(1, total + 1):
            if i not in seen:
                order.append(i)
        return order
