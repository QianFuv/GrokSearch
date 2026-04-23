"""Regression tests for configuration service behavior."""

import pytest

from grok_search.config import config
from grok_search.services.configuration import ConfigurationService


class FakeModelRegistry:
    """
    Minimal model registry stub for configuration service tests.
    """

    def __init__(self, models: list[str], should_fail: bool = False) -> None:
        """
        Store fake model validation state.

        Args:
            models: The fake model list returned by fetch operations.
            should_fail: Whether validation should fail.

        Returns:
            None.
        """
        self.models = models
        self.should_fail = should_fail

    async def fetch_available_models(self, api_url: str, api_key: str) -> list[str]:
        """
        Return the configured fake model list.

        Args:
            api_url: The upstream API URL.
            api_key: The upstream API key.

        Returns:
            The fake model list.
        """
        del api_url, api_key
        return self.models

    async def validate_model_selection(
        self, api_url: str, api_key: str, model: str
    ) -> None:
        """
        Validate a fake model selection.

        Args:
            api_url: The upstream API URL.
            api_key: The upstream API key.
            model: The requested model.

        Returns:
            None.

        Raises:
            ValueError: Raised when validation is configured to fail.
        """
        del api_url, api_key
        if self.should_fail or model not in self.models:
            raise ValueError(f"Invalid model: {model}")


@pytest.mark.asyncio
async def test_switch_model_rejects_unknown_model(monkeypatch) -> None:
    """
    Verify that unknown models are rejected before persistence.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(config, "_cached_model", "grok-4-fast", raising=False)

    saved: dict[str, str] = {}

    def fake_set_model(model: str) -> None:
        """
        Record the model that would have been persisted.

        Args:
            model: The model identifier to persist.

        Returns:
            None.
        """
        saved["model"] = model

    monkeypatch.setattr(config, "set_model", fake_set_model)

    result = await ConfigurationService(
        config,
        FakeModelRegistry(["grok-4-fast"], should_fail=True),
    ).switch_model("grok-4.1-thing")

    assert result["status"] == "failed"
    assert "Invalid model: grok-4.1-thing" in result["message"]
    assert "model" not in saved


@pytest.mark.asyncio
async def test_switch_model_accepts_validated_model(monkeypatch) -> None:
    """
    Verify that validated models are persisted successfully.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(config, "_cached_model", "grok-4-fast", raising=False)

    saved: dict[str, str] = {}

    def fake_set_model(model: str) -> None:
        """
        Record the validated model that would have been persisted.

        Args:
            model: The model identifier to persist.

        Returns:
            None.
        """
        saved["model"] = model
        config._cached_model = model

    monkeypatch.setattr(config, "set_model", fake_set_model)

    result = await ConfigurationService(
        config,
        FakeModelRegistry(["grok-4-fast", "grok-4.1-thing"]),
    ).switch_model("grok-4.1-thing")

    assert result["status"] == "success"
    assert result["current_model"] == "grok-4.1-thing"
    assert saved["model"] == "grok-4.1-thing"


@pytest.mark.asyncio
async def test_get_config_info_includes_connection_test(monkeypatch) -> None:
    """
    Verify that configuration info includes connection-test metadata.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("GROK_API_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setattr(config, "_cached_model", "grok-4-fast", raising=False)

    result = await ConfigurationService(
        config,
        FakeModelRegistry(["grok-4-fast", "grok-4.1-thing"]),
    ).get_config_info()

    assert result["connection_test"]["status"] == "connected"
    assert result["connection_test"]["available_models"] == [
        "grok-4-fast",
        "grok-4.1-thing",
    ]
