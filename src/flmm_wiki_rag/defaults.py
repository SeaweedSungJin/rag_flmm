"""Small helpers exposing default paths used by the CLI."""
from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "unified.yaml"
EXAMPLE_CONFIG_PATH = CONFIG_DIR / "unified.example.yaml"

__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DEFAULT_CONFIG_PATH",
    "EXAMPLE_CONFIG_PATH",
]
