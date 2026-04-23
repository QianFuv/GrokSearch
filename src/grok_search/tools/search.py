"""MCP registration for search and source-cache tools."""

from typing import Annotated

from fastmcp import FastMCP

from ..services import SearchService


def register_search_tools(mcp: FastMCP, search_service: SearchService) -> None:
    """
    Register search-related MCP tools on the provided FastMCP instance.

    Args:
        mcp: The FastMCP server instance.
        search_service: The search orchestration service.

    Returns:
        None.
    """

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
    ) -> dict[str, object]:
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
        return await search_service.web_search(query, platform, model, extra_sources)

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
    ) -> dict[str, object]:
        """
        Retrieve cached sources for a previous web_search session.

        Args:
            session_id: The session identifier returned by web_search.

        Returns:
            A dictionary containing the cached sources or an expiration error.
        """
        return await search_service.get_sources(session_id)
