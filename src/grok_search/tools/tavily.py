"""MCP registration for Tavily-backed fetch and map tools."""

import json
from typing import Annotated

import httpx
from fastmcp import Context, FastMCP
from pydantic import Field

from ..clients import TavilyClient
from ..config import Config
from ..logger import log_info


def register_tavily_tools(
    mcp: FastMCP,
    config: Config,
    tavily_client: TavilyClient,
) -> None:
    """
    Register Tavily-backed MCP tools on the provided FastMCP instance.

    Args:
        mcp: The FastMCP server instance.
        config: The runtime configuration instance.
        tavily_client: The Tavily client used by the tools.

    Returns:
        None.
    """

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
            - Ensure URL is complete and accessible (not behind authentication or
              paywalls).
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

        if not tavily_client.is_configured:
            return "Configuration error: TAVILY_API_KEY is not configured"

        result = await tavily_client.extract(url)
        if result:
            await log_info(ctx, "Fetch Finished (Tavily)!", config.debug_enabled)
            return result

        await log_info(ctx, "Fetch Failed!", config.debug_enabled)
        return "Extraction failed: Tavily did not return content after retries"

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
            Field(
                description="Maximum depth of mapping from the base URL.", ge=1, le=5
            ),
        ] = 1,
        max_breadth: Annotated[
            int,
            Field(
                description="Maximum number of links to follow per page.", ge=1, le=500
            ),
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
            Field(
                description="Maximum time in seconds for the operation.", ge=10, le=150
            ),
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
        try:
            result = await tavily_client.map_site(
                url=url,
                instructions=instructions,
                max_depth=max_depth,
                max_breadth=max_breadth,
                limit=limit,
                timeout=timeout,
            )
            return json.dumps(result, ensure_ascii=False, indent=2)
        except ValueError:
            return "Configuration error: TAVILY_API_KEY is not configured"
        except httpx.TimeoutException:
            return f"Map timeout: the request exceeded {timeout} seconds"
        except httpx.HTTPStatusError as error:
            return (
                f"HTTP error: {error.response.status_code} - "
                f"{error.response.text[:200]}"
            )
        except Exception as error:
            return f"Map error: {error}"
