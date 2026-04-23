"""Configuration diagnostics and model-switching helpers."""

from time import perf_counter
from typing import Any, Protocol

import httpx

from ..config import Config


class ModelRegistryProtocol(Protocol):
    """
    Describe the model-registry behavior required by the configuration service.
    """

    async def fetch_available_models(self, api_url: str, api_key: str) -> list[str]:
        """
        Fetch available models for the configured Grok endpoint.

        Args:
            api_url: The upstream API URL.
            api_key: The upstream API key.

        Returns:
            The available model identifiers.
        """

    async def validate_model_selection(
        self, api_url: str, api_key: str, model: str
    ) -> None:
        """
        Validate a requested model identifier.

        Args:
            api_url: The upstream API URL.
            api_key: The upstream API key.
            model: The requested model identifier.

        Returns:
            None.
        """


class ConfigurationService:
    """
    Expose configuration diagnostics and model switching helpers.

    Attributes:
        _config: The runtime configuration instance.
        _model_registry: The model registry used for connection tests.
    """

    def __init__(self, config: Config, model_registry: ModelRegistryProtocol) -> None:
        """
        Initialize the configuration service.

        Args:
            config: The runtime configuration instance.
            model_registry: The model registry used for connection tests.

        Returns:
            None.
        """
        self._config = config
        self._model_registry = model_registry

    async def get_config_info(self) -> dict[str, Any]:
        """
        Return current configuration details and connectivity diagnostics.

        Returns:
            A dictionary containing configuration details and connection status.
        """
        config_info = self._config.get_config_info()
        test_result: dict[str, object] = {
            "status": "not_tested",
            "message": "",
            "response_time_ms": 0,
        }

        try:
            start_time = perf_counter()
            model_names = await self._model_registry.fetch_available_models(
                self._config.grok_api_url,
                self._config.grok_api_key,
            )
            response_time = (perf_counter() - start_time) * 1000

            test_result["status"] = "connected"
            test_result["response_time_ms"] = round(response_time, 2)
            if model_names:
                test_result["message"] = (
                    "Fetched model list successfully; "
                    f"{len(model_names)} models available"
                )
                test_result["available_models"] = model_names
            else:
                test_result["message"] = (
                    "Fetched model list successfully, but the API returned no models"
                )
        except httpx.TimeoutException:
            test_result["status"] = "timeout"
            test_result["message"] = (
                "The request timed out after 10 seconds. "
                "Check network access or the API URL."
            )
        except httpx.RequestError as error:
            test_result["status"] = "request_error"
            test_result["message"] = f"Network error: {error}"
        except ValueError as error:
            test_result["status"] = "configuration_error"
            test_result["message"] = str(error)
        except Exception as error:
            test_result["status"] = "failed"
            test_result["message"] = f"Unexpected error: {error}"

        config_info["connection_test"] = test_result
        return config_info

    async def switch_model(self, model: str) -> dict[str, str]:
        """
        Persist a new default Grok model after validating it.

        Args:
            model: The model identifier to persist.

        Returns:
            A dictionary describing the update outcome.
        """
        try:
            previous_model = self._config.grok_model
            await self._model_registry.validate_model_selection(
                self._config.grok_api_url,
                self._config.grok_api_key,
                model,
            )
            self._config.set_model(model)
            current_model = self._config.grok_model
            return {
                "status": "success",
                "previous_model": previous_model,
                "current_model": current_model,
                "message": (
                    f"Switched the model from {previous_model} to {current_model}"
                ),
                "config_file": str(self._config.config_file),
            }
        except ValueError as error:
            return {
                "status": "failed",
                "message": f"Failed to switch models: {error}",
            }
        except Exception as error:
            return {
                "status": "failed",
                "message": f"Unexpected error: {error}",
            }
