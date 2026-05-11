"""Regression tests for packaging and test-runner configuration."""

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pytest_is_available_to_uv_run_without_extras() -> None:
    """Ensure plain `uv run pytest` syncs pytest into the project environment."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    dev_group = pyproject.get("dependency-groups", {}).get("dev", [])

    assert any(dep.startswith("pytest") for dep in dev_group)
    assert any(dep.startswith("pytest-asyncio") for dep in dev_group)
