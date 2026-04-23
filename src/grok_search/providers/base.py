"""Base provider abstractions used by the Grok Search server."""

from abc import ABC, abstractmethod


class BaseSearchProvider(ABC):
    """
    Define the interface required by search provider implementations.

    Attributes:
        api_url: The upstream API base URL.
        api_key: The credential used to authenticate requests.
    """

    def __init__(self, api_url: str, api_key: str) -> None:
        """
        Store the provider connection settings.

        Args:
            api_url: The upstream API base URL.
            api_key: The credential used to authenticate requests.

        Returns:
            None.
        """
        self.api_url = api_url
        self.api_key = api_key

    @abstractmethod
    async def search(
        self,
        query: str,
        platform: str = "",
        min_results: int = 3,
        max_results: int = 10,
        ctx=None,
    ) -> str:
        """
        Execute a provider-backed search request.

        Args:
            query: The user search query.
            platform: An optional platform filter.
            min_results: The minimum number of desired results.
            max_results: The maximum number of desired results.
            ctx: An optional MCP request context.

        Returns:
            A provider-specific response payload.
        """
        raise NotImplementedError

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Return the provider name used in logs and metadata.

        Returns:
            The provider name.
        """
        raise NotImplementedError
