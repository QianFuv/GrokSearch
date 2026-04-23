"""MCP registration for Grok-backed page and source-ranking tools."""

from typing import Annotated

from fastmcp import Context, FastMCP
from pydantic import Field

from ..services import GrokService


def register_content_tools(mcp: FastMCP, grok_service: GrokService) -> None:
    """
    Register Grok-backed content tools on the provided FastMCP instance.

    Args:
        mcp: The FastMCP server instance.
        grok_service: The Grok service used by the tools.

    Returns:
        None.
    """

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
    ) -> dict[str, str]:
        """
        Ask Grok to describe a single page and extract verbatim snippets.

        Args:
            url: The target page URL.
            model: An optional per-request model override.
            ctx: Optional FastMCP context used for logging.

        Returns:
            A dictionary containing the page description result.
        """
        return await grok_service.describe_url(url, model, ctx)

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
    ) -> dict[str, object]:
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
        return await grok_service.rank_sources(query, sources_text, total, model, ctx)
