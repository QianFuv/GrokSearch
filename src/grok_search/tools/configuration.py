"""MCP registration for configuration and model-management tools."""

import json
from typing import Annotated

from fastmcp import FastMCP

from ..services import ConfigurationService


def register_configuration_tools(
    mcp: FastMCP, configuration_service: ConfigurationService
) -> None:
    """
    Register configuration-related MCP tools on the provided FastMCP instance.

    Args:
        mcp: The FastMCP server instance.
        configuration_service: The configuration service used by the tools.

    Returns:
        None.
    """

    @mcp.tool(
        name="get_config_info",
        output_schema=None,
        description="""
        Returns current Grok Search MCP server configuration and tests API connectivity.

        **Key Features:**
            - **Configuration Check:** Verifies environment variables and current
              settings.
            - **Connection Test:** Sends request to /models endpoint to validate
              API access.
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
        return json.dumps(
            await configuration_service.get_config_info(),
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool(
        name="switch_model",
        output_schema=None,
        description="""
        Switches the default Grok model used for search and fetch operations,
        persisting the setting.

        **Key Features:**
            - **Model Selection:** Change the AI model for web search and content
              fetching.
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
        return json.dumps(
            await configuration_service.switch_model(model),
            ensure_ascii=False,
            indent=2,
        )
