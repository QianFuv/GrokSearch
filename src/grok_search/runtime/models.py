"""Model discovery, caching, and validation helpers."""

import asyncio

import httpx


class ModelRegistry:
    """
    Cache available Grok models and validate requested model identifiers.

    Attributes:
        _cache: Cached model identifiers keyed by endpoint and API key.
        _lock: The lock guarding cache access.
    """

    def __init__(self) -> None:
        """
        Initialize the in-memory model registry.

        Returns:
            None.
        """
        self._cache: dict[tuple[str, str], list[str]] = {}
        self._lock = asyncio.Lock()

    async def fetch_available_models(self, api_url: str, api_key: str) -> list[str]:
        """
        Fetch the available model identifiers from the upstream API.

        Args:
            api_url: The upstream API base URL.
            api_key: The API key used for authentication.

        Returns:
            A list of available model identifiers.
        """
        models_url = f"{api_url.rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                models_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        models: list[str] = []
        for item in (data or {}).get("data", []) or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(item["id"])
        return models

    async def get_available_models_cached(
        self, api_url: str, api_key: str
    ) -> list[str]:
        """
        Fetch available models with an in-memory cache.

        Args:
            api_url: The upstream API base URL.
            api_key: The API key used for authentication.

        Returns:
            A cached or freshly fetched list of model identifiers.
        """
        key = (api_url, api_key)
        async with self._lock:
            if key in self._cache:
                return self._cache[key]

        try:
            models = await self.fetch_available_models(api_url, api_key)
        except Exception:
            models = []

        async with self._lock:
            self._cache[key] = models
        return models

    @staticmethod
    def apply_model_suffix(api_url: str, model: str) -> str:
        """
        Normalize a model name for proxy-specific routing rules.

        Args:
            api_url: The upstream API base URL.
            model: The requested model identifier.

        Returns:
            The normalized model identifier.
        """
        if "openrouter" in api_url and ":online" not in model:
            return f"{model}:online"
        return model

    def is_available_model(
        self, api_url: str, requested_model: str, available_models: list[str]
    ) -> bool:
        """
        Check whether a requested model appears in the fetched model list.

        Args:
            api_url: The upstream API base URL.
            requested_model: The model requested by the caller.
            available_models: The fetched model identifiers.

        Returns:
            True when the requested model is available.
        """
        return any(
            candidate in available_models
            for candidate in (
                requested_model,
                self.apply_model_suffix(api_url, requested_model),
            )
        )

    async def resolve_request_model(
        self,
        api_url: str,
        api_key: str,
        requested_model: str,
        default_model: str,
    ) -> str:
        """
        Resolve the effective model for a request.

        Args:
            api_url: The upstream API base URL.
            api_key: The API key used for authentication.
            requested_model: The model explicitly requested by the caller.
            default_model: The configured default model identifier.

        Returns:
            The effective model identifier for the request.

        Raises:
            ValueError: Raised when an explicit model is invalid.
        """
        if not requested_model:
            return default_model

        available = await self.get_available_models_cached(api_url, api_key)
        if available and not self.is_available_model(
            api_url, requested_model, available
        ):
            raise ValueError(f"Invalid model: {requested_model}")
        return self.apply_model_suffix(api_url, requested_model)

    async def validate_model_selection(
        self, api_url: str, api_key: str, model: str
    ) -> None:
        """
        Validate that a model exists in the upstream /models response.

        Args:
            api_url: The upstream API base URL.
            api_key: The API key used for authentication.
            model: The model identifier to validate.

        Returns:
            None.

        Raises:
            ValueError: Raised when validation fails or no models are returned.
        """
        try:
            available = await self.fetch_available_models(api_url, api_key)
        except Exception as exc:
            raise ValueError(
                "Unable to validate the model against /models: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not available:
            raise ValueError(
                "Unable to validate the model because /models returned no data"
            )

        if not self.is_available_model(api_url, model, available):
            preview = ", ".join(available[:10])
            suffix = ", ..." if len(available) > 10 else ""
            raise ValueError(
                f"Invalid model: {model}. Available models: {preview}{suffix}"
            )
