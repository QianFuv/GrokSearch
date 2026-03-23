"""Configuration helpers for the Grok Search MCP server."""

import json
import os
from pathlib import Path
from typing import Any


class Config:
    """
    Provide singleton access to environment-backed server configuration.

    Attributes:
        _instance: The process-wide singleton instance.
        _config_file: The resolved configuration file path cache.
        _cached_model: The cached effective model identifier.
    """

    _instance: "Config | None" = None
    _config_file: Path | None
    _cached_model: str | None
    _SETUP_COMMAND = (
        "claude mcp add-json grok-search --scope user "
        '\'{"type":"stdio","command":"uvx","args":["--from",'
        '"git+https://github.com/GuDaStudio/GrokSearch","grok-search"],'
        '"env":{"GUDA_API_KEY":"your-guda-api-key"}}\''
    )
    _DEFAULT_MODEL = "grok-4.20-beta"
    _DEFAULT_GUDA_BASE_URL = "https://code.guda.studio"

    def __new__(cls) -> "Config":
        """
        Return the shared configuration singleton.

        Args:
            cls: The configuration class.

        Returns:
            The shared configuration instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config_file = None
            cls._instance._cached_model = None
        return cls._instance

    @property
    def config_file(self) -> Path:
        """
        Resolve the persisted configuration file location.

        Returns:
            The path to the configuration file.
        """
        if self._config_file is None:
            config_dir = Path.home() / ".config" / "grok-search"
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                config_dir = Path.cwd() / ".grok-search"
                config_dir.mkdir(parents=True, exist_ok=True)
            self._config_file = config_dir / "config.json"
        return self._config_file

    def _load_config_file(self) -> dict[str, Any]:
        """
        Load persisted configuration overrides from disk.

        Returns:
            A dictionary of persisted configuration values.
        """
        if not self.config_file.exists():
            return {}
        try:
            with open(self.config_file, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_config_file(self, config_data: dict[str, Any]) -> None:
        """
        Persist configuration overrides to disk.

        Args:
            config_data: The configuration data to write.

        Returns:
            None.

        Raises:
            ValueError: Raised when the configuration file cannot be written.
        """
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            raise ValueError(f"Unable to save config file: {e}") from e

    @property
    def debug_enabled(self) -> bool:
        """
        Read whether verbose debug logging is enabled.

        Returns:
            True when debug logging is enabled.
        """
        return os.getenv("GROK_DEBUG", "false").lower() in ("true", "1", "yes")

    @property
    def retry_max_attempts(self) -> int:
        """
        Read the maximum number of retry attempts for upstream calls.

        Returns:
            The configured retry attempt count.
        """
        return int(os.getenv("GROK_RETRY_MAX_ATTEMPTS", "3"))

    @property
    def retry_multiplier(self) -> float:
        """
        Read the exponential backoff multiplier for retries.

        Returns:
            The retry multiplier.
        """
        return float(os.getenv("GROK_RETRY_MULTIPLIER", "1"))

    @property
    def retry_max_wait(self) -> int:
        """
        Read the retry backoff cap in seconds.

        Returns:
            The maximum retry wait time.
        """
        return int(os.getenv("GROK_RETRY_MAX_WAIT", "10"))

    @property
    def guda_base_url(self) -> str:
        """
        Read the GuDa proxy base URL.

        Returns:
            The GuDa base URL.
        """
        return os.getenv("GUDA_BASE_URL", self._DEFAULT_GUDA_BASE_URL)

    @property
    def guda_api_key(self) -> str | None:
        """
        Read the GuDa API key when configured.

        Returns:
            The GuDa API key or None.
        """
        return os.getenv("GUDA_API_KEY")

    @property
    def grok_api_url(self) -> str:
        """
        Resolve the effective Grok API base URL.

        Returns:
            The Grok API base URL.

        Raises:
            ValueError: Raised when no Grok endpoint is configured.
        """
        url = os.getenv("GROK_API_URL")
        if not url:
            if self.guda_api_key:
                return f"{self.guda_base_url}/grok/v1"
            raise ValueError(
                "Grok API URL is not configured.\n"
                "Configure the MCP server with:\n"
                f"{self._SETUP_COMMAND}"
            )
        return url

    @property
    def grok_api_key(self) -> str:
        """
        Resolve the effective Grok API key.

        Returns:
            The Grok API key.

        Raises:
            ValueError: Raised when no Grok credential is configured.
        """
        key = os.getenv("GROK_API_KEY") or self.guda_api_key
        if not key:
            raise ValueError(
                "Grok API key is not configured.\n"
                "Configure the MCP server with:\n"
                f"{self._SETUP_COMMAND}"
            )
        return key

    @property
    def tavily_enabled(self) -> bool:
        """
        Read whether Tavily-backed helpers are enabled.

        Returns:
            True when Tavily helpers are enabled.
        """
        return os.getenv("TAVILY_ENABLED", "true").lower() in ("true", "1", "yes")

    @property
    def tavily_api_url(self) -> str:
        """
        Resolve the effective Tavily API base URL.

        Returns:
            The Tavily API base URL.
        """
        url = os.getenv("TAVILY_API_URL")
        if not url and self.guda_api_key:
            return f"{self.guda_base_url}/tavily"
        return url or "https://api.tavily.com"

    @property
    def tavily_api_key(self) -> str | None:
        """
        Resolve the effective Tavily API key.

        Returns:
            The Tavily API key or None.
        """
        return os.getenv("TAVILY_API_KEY") or self.guda_api_key

    @property
    def log_level(self) -> str:
        """
        Read the configured Python logging level name.

        Returns:
            The logging level name.
        """
        return os.getenv("GROK_LOG_LEVEL", "INFO").upper()

    @property
    def log_dir(self) -> Path:
        """
        Resolve the writable log directory path.

        Returns:
            The directory used for log files.
        """
        log_dir_str = os.getenv("GROK_LOG_DIR", "logs")
        log_dir = Path(log_dir_str)
        if log_dir.is_absolute():
            return log_dir

        home_log_dir = Path.home() / ".config" / "grok-search" / log_dir_str
        try:
            home_log_dir.mkdir(parents=True, exist_ok=True)
            return home_log_dir
        except OSError:
            pass

        cwd_log_dir = Path.cwd() / log_dir_str
        try:
            cwd_log_dir.mkdir(parents=True, exist_ok=True)
            return cwd_log_dir
        except OSError:
            pass

        tmp_log_dir = Path("/tmp") / "grok-search" / log_dir_str
        tmp_log_dir.mkdir(parents=True, exist_ok=True)
        return tmp_log_dir

    def _apply_model_suffix(self, model: str) -> str:
        """
        Normalize the model name for proxy-specific routing rules.

        Args:
            model: The raw configured model name.

        Returns:
            The normalized model name.
        """
        try:
            url = self.grok_api_url
        except ValueError:
            return model
        if "openrouter" in url and ":online" not in model:
            return f"{model}:online"
        return model

    @property
    def grok_model(self) -> str:
        """
        Resolve the effective Grok model identifier.

        Returns:
            The normalized Grok model name.
        """
        if self._cached_model is not None:
            return self._cached_model

        model = (
            os.getenv("GROK_MODEL")
            or self._load_config_file().get("model")
            or self._DEFAULT_MODEL
        )
        self._cached_model = self._apply_model_suffix(model)
        assert self._cached_model is not None
        return self._cached_model

    def set_model(self, model: str) -> None:
        """
        Persist a new default Grok model selection.

        Args:
            model: The new model identifier.

        Returns:
            None.
        """
        config_data = self._load_config_file()
        config_data["model"] = model
        self._save_config_file(config_data)
        self._cached_model = self._apply_model_suffix(model)

    @staticmethod
    def _mask_api_key(key: str) -> str:
        """
        Mask an API key while keeping the first and last four characters.

        Args:
            key: The API key to mask.

        Returns:
            The masked API key.
        """
        if not key or len(key) <= 8:
            return "***"
        return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"

    def get_config_info(self) -> dict[str, Any]:
        """
        Return configuration information with masked credentials.

        Returns:
            A dictionary describing the effective runtime configuration.
        """
        try:
            api_url = self.grok_api_url
            api_key_raw = self.grok_api_key
            api_key_masked = self._mask_api_key(api_key_raw)
            config_status = "configured"
        except ValueError as e:
            api_url = "not configured"
            api_key_masked = "not configured"
            config_status = f"configuration error: {e}"

        info = {
            "GUDA_BASE_URL": self.guda_base_url,
            "GUDA_API_KEY": self._mask_api_key(self.guda_api_key)
            if self.guda_api_key
            else "not configured",
            "GROK_API_URL": api_url,
            "GROK_API_KEY": api_key_masked,
            "GROK_MODEL": self.grok_model,
            "GROK_DEBUG": self.debug_enabled,
            "GROK_LOG_LEVEL": self.log_level,
            "GROK_LOG_DIR": str(self.log_dir),
            "TAVILY_API_URL": self.tavily_api_url,
            "TAVILY_ENABLED": self.tavily_enabled,
            "TAVILY_API_KEY": self._mask_api_key(self.tavily_api_key)
            if self.tavily_api_key
            else "not configured",
            "config_status": config_status,
        }
        return info


config = Config()
