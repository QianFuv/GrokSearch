"""Service helpers for Grok-backed tool operations."""

from ..config import Config
from ..providers.grok import GrokSearchProvider
from ..runtime import ModelRegistry, get_retry_settings


class GrokService:
    """
    Coordinate Grok provider creation and Grok-backed tool operations.

    Attributes:
        _config: The runtime configuration instance.
        _model_registry: The model registry used for validation and resolution.
    """

    def __init__(self, config: Config, model_registry: ModelRegistry) -> None:
        """
        Initialize the Grok service.

        Args:
            config: The runtime configuration instance.
            model_registry: The model registry used for validation and resolution.

        Returns:
            None.
        """
        self._config = config
        self._model_registry = model_registry

    def _create_provider(self, model: str) -> GrokSearchProvider:
        """
        Build a configured Grok provider instance.

        Args:
            model: The effective model identifier to use.

        Returns:
            A configured Grok provider instance.
        """
        retry_settings = get_retry_settings(self._config)
        return GrokSearchProvider(
            api_url=self._config.grok_api_url,
            api_key=self._config.grok_api_key,
            model=model,
            retry_settings=retry_settings,
            debug_enabled=self._config.debug_enabled,
        )

    async def resolve_request_model(self, requested_model: str) -> str:
        """
        Resolve the effective model for a Grok request.

        Args:
            requested_model: The optional per-request model override.

        Returns:
            The effective model identifier for the request.
        """
        return await self._model_registry.resolve_request_model(
            api_url=self._config.grok_api_url,
            api_key=self._config.grok_api_key,
            requested_model=requested_model,
            default_model=self._config.grok_model,
        )

    async def search(self, query: str, platform: str = "", model: str = "") -> str:
        """
        Execute a Grok-backed web search request.

        Args:
            query: The natural-language search query.
            platform: An optional platform focus hint.
            model: An optional per-request model override.

        Returns:
            The raw provider response text.
        """
        provider = self._create_provider(await self.resolve_request_model(model))
        return await provider.search(query, platform)

    async def describe_url(self, url: str, model: str = "", ctx=None) -> dict[str, str]:
        """
        Ask Grok to inspect a URL and return a title with extracts.

        Args:
            url: The target page URL.
            model: An optional per-request model override.
            ctx: Optional FastMCP context used for logging.

        Returns:
            A dictionary containing the page title, extracts, and URL.
        """
        try:
            provider = self._create_provider(await self.resolve_request_model(model))
        except ValueError as error:
            return {"url": url, "title": url, "extracts": "", "error": str(error)}

        try:
            return await provider.describe_url(url, ctx)
        except Exception as error:
            return {
                "url": url,
                "title": url,
                "extracts": "",
                "error": f"Describe URL failed: {type(error).__name__}: {error}",
            }

    async def rank_sources(
        self,
        query: str,
        sources_text: str,
        total: int,
        model: str = "",
        ctx=None,
    ) -> dict[str, object]:
        """
        Ask Grok to rank numbered sources by relevance to a query.

        Args:
            query: The user query.
            sources_text: The numbered source list.
            total: The total number of numbered sources.
            model: An optional per-request model override.
            ctx: Optional FastMCP context used for logging.

        Returns:
            A dictionary containing the ranked order or an error.
        """
        try:
            provider = self._create_provider(await self.resolve_request_model(model))
        except ValueError as error:
            return {"query": query, "order": [], "error": str(error)}

        try:
            order = await provider.rank_sources(query, sources_text, total, ctx)
        except Exception as error:
            return {
                "query": query,
                "order": [],
                "error": f"Rank sources failed: {type(error).__name__}: {error}",
            }
        return {"query": query, "order": order, "total": total}
