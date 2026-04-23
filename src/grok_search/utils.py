"""Shared text utility helpers for the Grok Search server."""

import re

_URL_PATTERN = re.compile(r'https?://[^\s<>"\'`，。、；：！？》）】\)]+')
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_META_REFUSAL_MARKERS = (
    "system instruction",
    "system instructions",
    'custom "system"',
    "custom system",
    "injected",
    "jailbreak",
    "override my behavior",
    "override my core",
    "built by xai",
)


def extract_unique_urls(text: str) -> list[str]:
    """
    Extract unique URLs from text in first-seen order.

    Args:
        text: The text to scan for URLs.

    Returns:
        A list of unique URLs in first-seen order.
    """
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_PATTERN.finditer(text):
        url = match.group().rstrip(".,;:!?")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def looks_like_meta_refusal(paragraph: str) -> bool:
    """
    Detect whether a paragraph is prompt-conflict boilerplate.

    Args:
        paragraph: A single response paragraph.

    Returns:
        True when the paragraph looks like a meta refusal instead of an answer.
    """
    normalized = " ".join((paragraph or "").lower().split())
    return any(marker in normalized for marker in _META_REFUSAL_MARKERS)


def sanitize_model_output(text: str) -> str:
    """
    Remove hidden reasoning and prompt-conflict boilerplate from model output.

    Args:
        text: The raw model response text.

    Returns:
        The cleaned response text.
    """
    cleaned = (text or "").replace("\r\n", "\n")
    cleaned = _THINK_BLOCK_PATTERN.sub("", cleaned).strip()
    if not cleaned:
        return ""

    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()
    ]
    while len(paragraphs) > 1 and looks_like_meta_refusal(paragraphs[0]):
        paragraphs.pop(0)

    if len(paragraphs) == 1 and looks_like_meta_refusal(paragraphs[0]):
        return ""

    cleaned = "\n\n".join(paragraphs).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)
