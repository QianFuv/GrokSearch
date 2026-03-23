"""Base search provider abstractions used by the Grok Search server."""

from abc import ABC, abstractmethod


class SearchResult:
    """
    Represent a normalized search result returned by a provider.

    Attributes:
        title: The result title.
        url: The canonical result URL.
        snippet: The summary text for the result.
        source: The source label or publisher.
        published_date: The publication date string when available.
    """

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: str = "",
        published_date: str = "",
    ) -> None:
        """
        Initialize a normalized search result object.

        Args:
            title: The result title.
            url: The canonical result URL.
            snippet: The summary text for the result.
            source: The source label or publisher.
            published_date: The publication date string when available.

        Returns:
            None.
        """
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.published_date = published_date

    def to_dict(self) -> dict[str, str]:
        """
        Convert the result into a JSON-serializable dictionary.

        Returns:
            A dictionary containing the normalized result fields.
        """
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date,
        }


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
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Return the provider name used in logs and metadata.

        Returns:
            The provider name.
        """
        pass
