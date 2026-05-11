"""
Grok provider implementations for search and page-level tools.
"""

import json
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.wait import wait_base

from ..config import config
from ..logger import log_info
from ..utils import (
    fetch_prompt,
    rank_sources_prompt,
    sanitize_model_output,
    search_prompt,
    url_describe_prompt,
)
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


def _strip_markdown_label(value: str) -> str:
    """
    Remove common Markdown list/heading/label decoration from a line.

    Args:
        value: The text to clean.

    Returns:
        Cleaned text suitable for a title, summary, or extract.
    """
    cleaned = value.strip()
    cleaned = re.sub(r"^[-*+]\s+", "", cleaned)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned)
    cleaned = cleaned.strip(" *_`\t")
    return cleaned.strip()


def _split_label_value(line: str) -> tuple[str, str] | None:
    """
    Split loose Markdown label lines such as '**Title:** Foo'.

    Args:
        line: A single model-output line.

    Returns:
        A lower-cased label and cleaned value, or None if no label exists.
    """
    match = re.match(
        r"^\s*(?:[-*+]\s*)?(?:\*\*)?([A-Za-z][A-Za-z ]{1,30})(?:\*\*)?\s*:\s*(.+?)\s*$",
        line,
    )
    if not match:
        return None
    return match.group(1).strip().lower(), _strip_markdown_label(match.group(2))


def _quoted_fragments(text: str) -> list[str]:
    """
    Extract double-quoted fragments and normalize quote marks.

    Args:
        text: Text that may contain quoted excerpts.

    Returns:
        A list of fragments wrapped in ASCII double quotes.
    """
    fragments = re.findall(r'["“]([^"”]+)["”]', text)
    return [f'"{fragment.strip()}"' for fragment in fragments]


def _append_extracts(extracts: list[str], text: str) -> None:
    """
    Append extract fragments from text while preferring quoted verbatim snippets.

    Args:
        extracts: The extract accumulator to mutate.
        text: A candidate extract line or paragraph.

    Returns:
        None.
    """
    cleaned = _strip_markdown_label(text)
    if not cleaned:
        return
    quoted = _quoted_fragments(cleaned)
    if quoted:
        extracts.extend(quoted)
    else:
        extracts.append(cleaned)


def _natural_language_description_parts(text: str) -> dict[str, str]:
    """
    Pull title, summary, and extracts from prose-like model output.

    Args:
        text: The full model output.

    Returns:
        A partial dictionary with any parsed fields.
    """
    fields: dict[str, str] = {}
    title_match = re.search(
        r"\b(?:page\s+)?title\s+(?:is|=|:)\s*[\"“]([^\"”]+)[\"”]",
        text,
        re.IGNORECASE,
    )
    if title_match:
        fields["title"] = title_match.group(1).strip()

    summary_match = re.search(
        r"\b(?:summary|description)\s*:\s*(.*?)(?=\s+"
        r"(?:extracts?|quotes?)\s*(?:include|:)|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if summary_match:
        fields["summary"] = summary_match.group(1).strip().rstrip(" .") + "."

    extracts_match = re.search(
        r"\b(?:extracts?|quotes?)\s*(?:include|:)\s*(.+)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if extracts_match:
        fragments = _quoted_fragments(extracts_match.group(1))
        if fragments:
            fields["extracts"] = " | ".join(fragments)

    return fields


def _parse_url_description_result(raw_result: str, url: str) -> dict[str, str]:
    """
    Parse exact, loose Markdown, or prose URL-description model output.

    Args:
        raw_result: The model output returned by the provider.
        url: The URL being described.

    Returns:
        A dictionary with title, summary, extracts, and url fields.
    """
    result = sanitize_model_output(raw_result).strip()
    natural_fields = _natural_language_description_parts(result)
    title = natural_fields.get("title", "")
    summary = natural_fields.get("summary", "")
    extracts: list[str] = []
    if natural_fields.get("extracts"):
        extracts.extend(natural_fields["extracts"].split(" | "))

    mode = ""
    summary_lines: list[str] = []
    for raw_line in result.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        label_value = _split_label_value(line)
        if label_value:
            label, value = label_value
            if label in {"title", "page title"}:
                title = value or title
                mode = ""
                continue
            if label in {"summary", "description"}:
                summary = value or summary
                mode = ""
                continue
            if label in {"extract", "extracts", "quotes", "key extracts"}:
                _append_extracts(extracts, value)
                mode = "extracts"
                continue

        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading:
            heading_text = _strip_markdown_label(heading.group(1))
            if "extract" in heading_text.lower() or "quote" in heading_text.lower():
                mode = "extracts"
            elif not title:
                title = heading_text
                mode = ""
            else:
                mode = ""
            continue

        if mode == "extracts" or re.match(r"^[-*+]\s+", line):
            _append_extracts(extracts, line)
            continue

        if not summary and not summary_lines:
            summary_lines.append(_strip_markdown_label(line))

    if not title:
        title = url
    if not summary and summary_lines:
        summary = " ".join(line for line in summary_lines if line).strip()

    deduped_extracts: list[str] = []
    seen_extracts: set[str] = set()
    for extract in extracts:
        if extract and extract not in seen_extracts:
            seen_extracts.add(extract)
            deduped_extracts.append(extract)

    response = {
        "title": title,
        "summary": summary,
        "extracts": " | ".join(deduped_extracts),
        "url": url,
    }
    if not summary and not deduped_extracts:
        response["summary"] = result
    return response


def _describe_url_failure(url: str, message: str) -> dict[str, str]:
    """
    Build a stable fallback response for describe_url failures.

    Args:
        url: The URL being described.
        message: The failure detail.

    Returns:
        A diagnostic describe_url response.
    """
    return {
        "title": url,
        "summary": f"Unable to describe URL: {message}",
        "extracts": "",
        "url": url,
        "error": f"Describe URL failed: {message}",
    }


def _join_api_url(base_url: str, path: str, api_version: str = "v1") -> str:
    """
    Join an OpenAI-compatible API base URL with an endpoint path.

    Args:
        base_url: The configured API base URL.
        path: The endpoint path to append.
        api_version: The version segment to add when missing.

    Returns:
        The normalized endpoint URL.
    """
    normalized_base = base_url.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    version_fragment = f"/{api_version}"

    if normalized_base.endswith(version_fragment):
        return f"{normalized_base}{normalized_path}"

    return f"{normalized_base}{version_fragment}{normalized_path}"


def _extract_responses_text(payload: dict[str, Any]) -> str:
    """
    Extract final message text from a Responses API payload.

    Args:
        payload: A decoded JSON payload from the upstream API.

    Returns:
        The extracted text, or an empty string when unavailable.
    """
    if not isinstance(payload, dict):
        return ""

    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    fragments: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in _get_responses_message_content(item):
            if not isinstance(block, dict) or block.get("type") != "output_text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text:
                fragments.append(text)

    return "\n".join(fragments)


def _normalize_source_title(title: str | None, url: str) -> str:
    """
    Normalize a source title into a readable label.

    Args:
        title: The raw source title.
        url: The source URL.

    Returns:
        A normalized title suitable for markdown source listings.
    """
    clean_title = (title or "").strip()
    if clean_title and not clean_title.isdigit():
        return clean_title

    return urlparse(url).hostname or url


def _get_responses_message_content(item: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract message content blocks from a Responses output item.

    Args:
        item: A single Responses output item.

    Returns:
        The normalized list of message content blocks.
    """
    content = item.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]

    message = item.get("message")
    if isinstance(message, dict):
        nested_content = message.get("content")
        if isinstance(nested_content, list):
            return [block for block in nested_content if isinstance(block, dict)]

    return []


def _extract_responses_sources(payload: dict[str, Any]) -> list[dict[str, str]]:
    """
    Extract and deduplicate sources from a Responses API payload.

    Args:
        payload: A decoded JSON payload from the upstream API.

    Returns:
        A first-seen ordered list of normalized sources.
    """
    if not isinstance(payload, dict):
        return []

    output = payload.get("output")
    if not isinstance(output, list):
        return []

    seen: set[str] = set()
    sources: list[dict[str, str]] = []

    def _add_source(url: str, title: str | None) -> None:
        if not url or url in seen:
            return
        seen.add(url)
        sources.append({"title": _normalize_source_title(title, url), "url": url})

    for item in output:
        if not isinstance(item, dict):
            continue

        if item.get("type") == "web_search_call":
            action = item.get("action")
            if isinstance(action, dict):
                raw_sources = action.get("sources")
                if isinstance(raw_sources, list):
                    for source in raw_sources:
                        if isinstance(source, dict):
                            url = source.get("url")
                            if isinstance(url, str):
                                _add_source(url, source.get("title"))
            continue

        if item.get("type") != "message":
            continue

        for block in _get_responses_message_content(item):
            annotations = block.get("annotations")
            if not isinstance(annotations, list):
                continue
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                if annotation.get("type") != "url_citation":
                    continue
                url = annotation.get("url")
                if isinstance(url, str):
                    _add_source(url, annotation.get("title"))

    return sources


def _format_sources_markdown(sources: list[dict[str, str]]) -> str:
    """
    Format normalized sources into the provider's trailing markdown block.

    Args:
        sources: The normalized source list.

    Returns:
        A markdown sources section, or an empty string.
    """
    if not sources:
        return ""

    lines = ["## Sources"]
    for source in sources:
        lines.append(f"- [{source['title']}]({source['url']})")
    return "\n\n" + "\n".join(lines)


def _is_responses_payload(payload: Any) -> bool:
    """
    Check whether a decoded payload looks like a Responses API body.

    Args:
        payload: The decoded payload.

    Returns:
        True when the payload matches the expected Responses shape.
    """
    return isinstance(payload, dict) and isinstance(payload.get("output"), list)


def _extract_responses_payload_from_sse(body: str) -> dict[str, Any] | None:
    """
    Extract the final Responses payload from an SSE response body.

    Some OpenAI-compatible Responses facades return ``text/event-stream`` even
    for non-streaming requests. Prefer the final ``response.completed`` event,
    and fall back to constructing a minimal Responses-shaped payload from text
    deltas when the final event is absent.
    """
    completed_payload: dict[str, Any] | None = None
    text_fragments: list[str] = []
    done_text = ""
    done_annotations: list[dict[str, Any]] = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload_text = line[5:].lstrip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            event_payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue

        if _is_responses_payload(event_payload):
            completed_payload = event_payload
            continue

        response_payload = event_payload.get("response")
        if _is_responses_payload(response_payload):
            completed_payload = response_payload
            continue

        event_type = event_payload.get("type")
        if event_type == "response.output_text.delta":
            delta = event_payload.get("delta")
            if isinstance(delta, str):
                text_fragments.append(delta)
        elif event_type == "response.output_text.done":
            text = event_payload.get("text")
            if isinstance(text, str):
                done_text = text
            annotations = event_payload.get("annotations")
            if isinstance(annotations, list):
                done_annotations = [a for a in annotations if isinstance(a, dict)]

    if completed_payload is not None:
        return completed_payload

    text = done_text or "".join(text_fragments)
    if not text:
        return None

    return {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": done_annotations,
                    }
                ],
            }
        ]
    }


def _looks_like_responses_unsupported_error(response: httpx.Response) -> bool:
    """
    Check whether an HTTP error indicates the Responses API is unsupported.

    Args:
        response: The upstream HTTP response.

    Returns:
        True when the error suggests this backend does not support Responses.
    """
    if response.status_code in {404, 405}:
        return True

    if response.status_code not in {400, 422}:
        return False

    try:
        body = json.dumps(response.json(), ensure_ascii=False).lower()
    except ValueError:
        body = response.text.lower()

    unsupported_markers = (
        "unsupported",
        "not supported",
        "unknown parameter",
        "unsupported parameter",
        "unexpected parameter",
        "unknown field",
        "invalid tool",
        "unknown tool",
        "extra inputs are not permitted",
    )
    responses_markers = (
        "responses",
        "/responses",
        "tools",
        "web_search",
        "instructions",
        "input",
    )
    return any(marker in body for marker in unsupported_markers) and any(
        marker in body for marker in responses_markers
    )


def _looks_like_duplicate_web_search_tools_error(response: httpx.Response) -> bool:
    """
    Check whether the backend rejected an explicit web_search tool as duplicate.

    Args:
        response: The upstream HTTP response.

    Returns:
        True when the backend wants Responses search without explicit tools.
    """
    if response.status_code not in {400, 422}:
        return False

    try:
        body = json.dumps(response.json(), ensure_ascii=False).lower()
    except ValueError:
        body = response.text.lower()

    return "multiple web search tools" in body and "not supported" in body


def _responses_payload_without_tools(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Copy a Responses payload while removing explicit tool declarations.

    Args:
        payload: The original Responses payload.

    Returns:
        A shallow payload copy without the tools field.
    """
    toolless_payload = dict(payload)
    toolless_payload.pop("tools", None)
    return toolless_payload


class ResponsesUnsupportedError(RuntimeError):
    """
    Raised when an upstream backend clearly does not support Responses search.
    """


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

    def __init__(self, api_url: str, api_key: str, model: str = "grok-4-fast"):
        """
        Initialize the Grok search provider.

        Args:
            api_url: The upstream API base URL.
            api_key: The API key used for authentication.
            model: The model identifier used for requests.

        Returns:
            None.
        """
        super().__init__(api_url, api_key)
        self.model = model

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
        full_query = time_context + query + platform_prompt

        responses_payload = {
            "model": self.model,
            "instructions": search_prompt,
            "input": [{"role": "user", "content": full_query}],
            "tools": [{"type": "web_search"}],
        }
        legacy_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": search_prompt,
                },
                {"role": "user", "content": full_query},
            ],
            "stream": True,
        }

        await log_info(
            ctx, f"platform_prompt: {query + platform_prompt}", config.debug_enabled
        )

        try:
            return await self._execute_responses_search_with_retry(
                headers, responses_payload, ctx
            )
        except ResponsesUnsupportedError:
            return await self._execute_stream_with_retry(headers, legacy_payload, ctx)

    async def fetch(self, url: str, ctx=None) -> str:
        """
        Fetch and convert a webpage into structured Markdown text.

        Args:
            url: The target page URL.
            ctx: Optional FastMCP context used for logging.

        Returns:
            The raw model response text.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": fetch_prompt,
                },
                {
                    "role": "user",
                    "content": url + "\n获取该网页内容并返回其结构化Markdown格式",
                },
            ],
            "stream": True,
        }
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

        await log_info(ctx, f"content: {content}", config.debug_enabled)

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
                stop=stop_after_attempt(config.retry_max_attempts + 1),
                wait=_WaitWithRetryAfter(
                    config.retry_multiplier, config.retry_max_wait
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

    async def _execute_responses_search_with_retry(
        self, headers: dict[str, str], payload: dict[str, Any], ctx=None
    ) -> str:
        """
        Execute a non-streaming Responses search request with retry handling.

        Args:
            headers: The HTTP headers for the request.
            payload: The JSON payload sent to the upstream API.
            ctx: Optional FastMCP context used for logging.

        Returns:
            The cleaned model response text with an optional sources block.
        """
        timeout = httpx.Timeout(connect=6.0, read=120.0, write=10.0, pool=None)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(config.retry_max_attempts + 1),
                wait=_WaitWithRetryAfter(
                    config.retry_multiplier, config.retry_max_wait
                ),
                retry=retry_if_exception(_is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    endpoint = _join_api_url(self.api_url, "/responses")
                    effective_payload = payload
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        json=effective_payload,
                    )
                    if (
                        "tools" in effective_payload
                        and _looks_like_duplicate_web_search_tools_error(response)
                    ):
                        await log_info(
                            ctx,
                            "Responses backend rejected explicit web_search "
                            "tools; retrying without tools",
                            config.debug_enabled,
                        )
                        effective_payload = _responses_payload_without_tools(
                            effective_payload
                        )
                        response = await client.post(
                            endpoint,
                            headers=headers,
                            json=effective_payload,
                        )

                    if _looks_like_responses_unsupported_error(response):
                        raise ResponsesUnsupportedError(
                            f"Responses API unsupported: HTTP {response.status_code}"
                        )

                    response.raise_for_status()

                    if "text/event-stream" in response.headers.get(
                        "content-type", ""
                    ).lower():
                        data = _extract_responses_payload_from_sse(response.text)
                        if data is None:
                            raise ResponsesUnsupportedError(
                                "Responses API returned an empty SSE payload"
                            )
                    else:
                        try:
                            data = response.json()
                        except ValueError as exc:
                            raise ResponsesUnsupportedError(
                                "Responses API returned a non-JSON payload"
                            ) from exc

                    if not _is_responses_payload(data):
                        raise ResponsesUnsupportedError(
                            "Responses API returned a non-Responses payload"
                        )

                    answer = sanitize_model_output(_extract_responses_text(data))
                    sources_block = _format_sources_markdown(
                        _extract_responses_sources(data)
                    )

                    if answer and sources_block:
                        result = f"{answer}{sources_block}"
                    elif sources_block:
                        result = sources_block.lstrip()
                    else:
                        result = answer

                    await log_info(ctx, f"content: {result}", config.debug_enabled)
                    return result

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
                {"role": "system", "content": url_describe_prompt},
                {"role": "user", "content": url},
            ],
            "stream": True,
        }
        try:
            result = await self._execute_stream_with_retry(headers, payload, ctx)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return _describe_url_failure(url, message)

        if not result or not result.strip():
            return _describe_url_failure(url, "empty provider response")

        return _parse_url_description_result(result, url)

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
                {"role": "system", "content": rank_sources_prompt},
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
