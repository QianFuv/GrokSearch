"""Regression tests for source parsing and normalization helpers."""

import pytest

from grok_search.sources import SourcesCache, merge_sources, split_answer_and_sources


def test_merge_sources_normalizes_deduplicates_and_preserves_provenance_order():
    primary = [
        {"title": "Primary", "url": " https://example.com/a ", "provider": "grok"},
        {"title": "Duplicate", "url": "https://example.com/a", "provider": "tavily"},
    ]
    extra = [
        {
            "title": "Extra",
            "url": "https://example.com/b",
            "content": "Extra snippet",
            "provider": "tavily",
        }
    ]

    assert merge_sources(primary, extra) == [
        {"title": "Primary", "url": "https://example.com/a", "provider": "grok"},
        {
            "title": "Extra",
            "url": "https://example.com/b",
            "description": "Extra snippet",
            "provider": "tavily",
        },
    ]


def test_split_answer_and_sources_deduplicates_inline_and_tail_sources_in_order():
    text = """
Answer cites [Alpha](https://example.com/a) inline.

Sources:
- [Alpha duplicate](https://example.com/a)
- [Beta](https://example.com/b)
"""

    answer, sources = split_answer_and_sources(text)

    assert answer == "Answer cites [Alpha](https://example.com/a) inline."
    assert sources == [
        {"title": "Alpha duplicate", "url": "https://example.com/a"},
        {"title": "Beta", "url": "https://example.com/b"},
    ]


def test_split_answer_and_sources_leaves_single_details_source_in_answer():
    text = """
Answer body.

<details>
<summary>Sources</summary>
- [Only](https://example.com/only)
</details>
"""

    answer, sources = split_answer_and_sources(text)

    assert answer == text.strip()
    assert sources == []


def test_split_answer_and_sources_leaves_single_tail_link_in_answer():
    text = "Answer body.\n\nhttps://example.com/only"

    answer, sources = split_answer_and_sources(text)

    assert answer == text
    assert sources == []


@pytest.mark.asyncio
async def test_sources_cache_eviction_uses_lru_order():
    cache = SourcesCache(max_size=2)
    await cache.set("old", [{"url": "https://example.com/old"}])
    await cache.set("kept", [{"url": "https://example.com/kept"}])

    assert await cache.get("old") == [{"url": "https://example.com/old"}]

    await cache.set("new", [{"url": "https://example.com/new"}])

    assert await cache.get("kept") is None
    assert await cache.get("old") == [{"url": "https://example.com/old"}]
    assert await cache.get("new") == [{"url": "https://example.com/new"}]
