"""FastMCP application assembly for the Grok Search server."""

from fastmcp import FastMCP

from .clients import TavilyClient
from .config import config
from .runtime import ModelRegistry
from .services import ConfigurationService, GrokService, SearchService
from .tools import (
    register_configuration_tools,
    register_content_tools,
    register_search_tools,
    register_tavily_tools,
)

model_registry = ModelRegistry()
tavily_client = TavilyClient(config)
grok_service = GrokService(config, model_registry)
search_service = SearchService(config, grok_service, tavily_client)
configuration_service = ConfigurationService(config, model_registry)

mcp = FastMCP("grok-search")

register_search_tools(mcp, search_service)
register_tavily_tools(mcp, config, tavily_client)
register_configuration_tools(mcp, configuration_service)
register_content_tools(mcp, grok_service)
