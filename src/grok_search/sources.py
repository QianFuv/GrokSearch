"""
Helpers for splitting answer text from trailing source listings.
"""

import ast
import asyncio
import json
import re
import uuid
from collections import OrderedDict
from typing import Any

from .utils import extract_unique_urls

_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_SOURCE_LABEL_PATTERN = (
    r"(sources?|references?|citations?|resources?|links?|further reading|read more|"
    r"信源|参考资料|参考|引用|来源列表|来源)"
)
_SOURCES_HEADING_PATTERN = re.compile(
    r"(?im)^"
    r"(?:#{1,6}\s*)?"
    r"(?:\*\*|__)?\s*" + _SOURCE_LABEL_PATTERN + r"\s*(?:\*\*|__)?"
    r"(?:\s*[（(][^)\n]*[)）])?"
    r"\s*[:：]?\s*$"
)
_INLINE_SOURCES_HEADING_PATTERN = re.compile(
    r"(?im)^"
    r"(?:#{1,6}\s*)?"
    r"(?:\*\*|__)?\s*" + _SOURCE_LABEL_PATTERN + r"\s*(?:\*\*|__)?"
    r"(?:\s*[（(][^)\n]*[)）])?"
    r"\s*[:：]\s+.+$"
)
_SOURCES_FUNCTION_PATTERN = re.compile(
    r"(?im)(^|\n)\s*(sources|source|citations|citation|references|reference|citation_card|source_cards|source_card)\s*\("
)
_FOOTNOTE_LINK_LINE_PATTERN = re.compile(r"^\[[^\]]+\]:\s*https?://\S+$")


def new_session_id() -> str:
    """
    Generate a short session identifier for cached search sources.

    Returns:
        A short random hexadecimal session identifier.
    """
    return uuid.uuid4().hex[:12]


class SourcesCache:
    """
    Store cached source lists in a bounded asynchronous LRU cache.

    Attributes:
        _max_size: The maximum number of cached sessions.
        _lock: The lock guarding cache mutations.
        _cache: The ordered mapping of session IDs to sources.
    """

    def __init__(self, max_size: int = 256):
        """
        Initialize the bounded source cache.

        Args:
            max_size: The maximum number of cached sessions to retain.

        Returns:
            None.
        """
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()

    async def set(self, session_id: str, sources: list[dict]) -> None:
        """
        Store sources for a session and evict the oldest entries when needed.

        Args:
            session_id: The session identifier.
            sources: The normalized source list to cache.

        Returns:
            None.
        """
        async with self._lock:
            self._cache[session_id] = sources
            self._cache.move_to_end(session_id)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    async def get(self, session_id: str) -> list[dict] | None:
        """
        Retrieve cached sources for a session.

        Args:
            session_id: The session identifier.

        Returns:
            The cached source list, or None when missing.
        """
        async with self._lock:
            sources = self._cache.get(session_id)
            if sources is None:
                return None
            self._cache.move_to_end(session_id)
            return sources


def merge_sources(*source_lists: list[dict]) -> list[dict]:
    """
    Merge multiple source lists while deduplicating by URL.

    Args:
        *source_lists: Source lists to merge.

    Returns:
        A merged list containing unique URLs in first-seen order.
    """
    seen: set[str] = set()
    merged: list[dict] = []
    for sources in source_lists:
        for item in sources or []:
            url = (item or {}).get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            url = url.strip()
            if url in seen:
                continue
            seen.add(url)
            merged.append(item)
    return merged


def split_answer_and_sources(text: str) -> tuple[str, list[dict]]:
    """
    Split an answer body from any trailing source listings.

    Args:
        text: The raw model output text.

    Returns:
        A tuple of cleaned answer text and normalized source records.
    """
    raw = (text or "").strip()
    if not raw:
        return "", []

    split = _split_function_call_sources(raw)
    if split:
        return split

    split = _split_heading_sources(raw)
    if split:
        return split

    split = _split_inline_heading_sources(raw)
    if split:
        return split

    split = _split_details_block_sources(raw)
    if split:
        return split

    split = _split_tail_link_block(raw)
    if split:
        return split

    return raw, []


def _split_inline_heading_sources(text: str) -> tuple[str, list[dict]] | None:
    """
    Split trailing source sections that keep URLs on the same heading line.

    Args:
        text: The raw model output text.

    Returns:
        A tuple of answer text and sources when a matching block is found.
    """
    matches = list(_INLINE_SOURCES_HEADING_PATTERN.finditer(text))
    if not matches:
        return None

    for match in reversed(matches):
        start = match.start()
        sources_text = text[start:]
        sources = _extract_sources_from_text(sources_text)
        if not sources:
            continue
        answer = text[:start].rstrip()
        return answer, sources

    return None


def _split_function_call_sources(text: str) -> tuple[str, list[dict]] | None:
    """
    Split source payloads expressed as trailing function-like calls.

    Args:
        text: The raw model output text.

    Returns:
        A tuple of answer text and sources when a matching payload is found.
    """
    matches = list(_SOURCES_FUNCTION_PATTERN.finditer(text))
    if not matches:
        return None

    for m in reversed(matches):
        open_paren_idx = m.end() - 1
        extracted = _extract_balanced_call_at_end(text, open_paren_idx)
        if not extracted:
            continue

        close_paren_idx, args_text = extracted
        sources = _parse_sources_payload(args_text)
        if not sources:
            continue

        answer = text[: m.start()].rstrip()
        return answer, sources

    return None


def _extract_balanced_call_at_end(
    text: str, open_paren_idx: int
) -> tuple[int, str] | None:
    """
    Extract the argument payload from a balanced trailing call expression.

    Args:
        text: The raw text containing the trailing call.
        open_paren_idx: The index of the opening parenthesis.

    Returns:
        A tuple of closing parenthesis index and argument text when valid.
    """
    if open_paren_idx < 0 or open_paren_idx >= len(text) or text[open_paren_idx] != "(":
        return None

    depth = 1
    in_string: str | None = None
    escape = False

    for idx in range(open_paren_idx + 1, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == in_string:
                in_string = None
            continue

        if ch in ("'", '"'):
            in_string = ch
            continue

        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                if text[idx + 1 :].strip():
                    return None
                args_text = text[open_paren_idx + 1 : idx]
                return idx, args_text

    return None


def _split_heading_sources(text: str) -> tuple[str, list[dict]] | None:
    """
    Split trailing source sections introduced by a dedicated heading line.

    Args:
        text: The raw model output text.

    Returns:
        A tuple of answer text and sources when a heading block is found.
    """
    matches = list(_SOURCES_HEADING_PATTERN.finditer(text))
    if not matches:
        return None

    for m in reversed(matches):
        start = m.start()
        sources_text = text[start:]
        sources = _extract_sources_from_text(sources_text)
        if not sources:
            continue
        answer = text[:start].rstrip()
        return answer, sources
    return None


def _split_tail_link_block(text: str) -> tuple[str, list[dict]] | None:
    """
    Split a trailing block made up only of source links.

    Args:
        text: The raw model output text.

    Returns:
        A tuple of answer text and sources when a trailing link block is found.
    """
    lines = text.splitlines()
    if not lines:
        return None

    idx = len(lines) - 1
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx < 0:
        return None

    tail_end = idx
    link_like_count = 0
    while idx >= 0:
        line = lines[idx].strip()
        if not line:
            idx -= 1
            continue
        if not _is_link_only_line(line):
            break
        link_like_count += 1
        idx -= 1

    tail_start = idx + 1
    if link_like_count < 2:
        return None

    block_text = "\n".join(lines[tail_start : tail_end + 1])
    sources = _extract_sources_from_text(block_text)
    if not sources:
        return None

    answer = "\n".join(lines[:tail_start]).rstrip()
    return answer, sources


def _split_details_block_sources(text: str) -> tuple[str, list[dict]] | None:
    """
    Split trailing sources wrapped inside an HTML details block.

    Args:
        text: The raw model output text.

    Returns:
        A tuple of answer text and sources when a supported block is found.
    """
    lower = text.lower()
    close_idx = lower.rfind("</details>")
    if close_idx == -1:
        return None
    tail = text[close_idx + len("</details>") :].strip()
    if tail:
        return None

    open_idx = lower.rfind("<details", 0, close_idx)
    if open_idx == -1:
        return None

    block_text = text[open_idx : close_idx + len("</details>")]
    sources = _extract_sources_from_text(block_text)
    if len(sources) < 2:
        return None

    answer = text[:open_idx].rstrip()
    return answer, sources


def _is_link_only_line(line: str) -> bool:
    """
    Check whether a line is effectively only a source link.

    Args:
        line: A single text line.

    Returns:
        True when the line represents a standalone source link.
    """
    stripped = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line).strip()
    if not stripped:
        return False
    if stripped.startswith(("http://", "https://")):
        return True
    if _FOOTNOTE_LINK_LINE_PATTERN.fullmatch(stripped):
        return True
    return bool(_MD_LINK_PATTERN.search(stripped))


def _parse_sources_payload(payload: str) -> list[dict]:
    """
    Parse a structured source payload from JSON, Python literals, or raw text.

    Args:
        payload: The payload string to parse.

    Returns:
        A normalized list of source dictionaries.
    """
    payload = (payload or "").strip().rstrip(";")
    if not payload:
        return []

    data: Any = None
    try:
        data = json.loads(payload)
    except Exception:
        try:
            data = ast.literal_eval(payload)
        except Exception:
            data = None

    if data is None:
        return _extract_sources_from_text(payload)

    if isinstance(data, dict):
        for key in ("sources", "citations", "references", "urls"):
            if key in data:
                return _normalize_sources(data[key])
        return _normalize_sources(data)

    return _normalize_sources(data)


def _normalize_sources(data: Any) -> list[dict]:
    """
    Normalize heterogeneous source payloads into URL-keyed dictionaries.

    Args:
        data: Raw source data from parsed payloads.

    Returns:
        A normalized list of source dictionaries.
    """
    items: list[Any]
    if isinstance(data, (list, tuple)):
        items = list(data)
    elif isinstance(data, dict):
        items = [data]
    else:
        items = [data]

    normalized: list[dict] = []
    seen: set[str] = set()

    for item in items:
        if isinstance(item, str):
            for url in extract_unique_urls(item):
                if url not in seen:
                    seen.add(url)
                    normalized.append({"url": url})
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, url = item[0], item[1]
            if (
                isinstance(url, str)
                and url.startswith(("http://", "https://"))
                and url not in seen
            ):
                seen.add(url)
                out: dict = {"url": url}
                if isinstance(title, str) and title.strip():
                    out["title"] = title.strip()
                normalized.append(out)
            continue

        if isinstance(item, dict):
            url_value = item.get("url") or item.get("href") or item.get("link")
            if not isinstance(url_value, str) or not url_value.startswith(
                ("http://", "https://")
            ):
                continue
            url = url_value
            if url in seen:
                continue
            seen.add(url)
            source_item: dict = {"url": url}
            title = item.get("title") or item.get("name") or item.get("label")
            if isinstance(title, str) and title.strip():
                source_item["title"] = title.strip()
            desc = item.get("description") or item.get("snippet") or item.get("content")
            if isinstance(desc, str) and desc.strip():
                source_item["description"] = desc.strip()
            normalized.append(source_item)
            continue

    return normalized


def _extract_sources_from_text(text: str) -> list[dict]:
    """
    Extract Markdown and plain URLs from free-form text.

    Args:
        text: The text to scan for source URLs.

    Returns:
        A normalized list of extracted source dictionaries.
    """
    sources: list[dict] = []
    seen: set[str] = set()

    for title, url in _MD_LINK_PATTERN.findall(text or ""):
        url = (url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = (title or "").strip()
        if title:
            sources.append({"title": title, "url": url})
        else:
            sources.append({"url": url})

    for url in extract_unique_urls(text or ""):
        if url in seen:
            continue
        seen.add(url)
        sources.append({"url": url})

    return sources
