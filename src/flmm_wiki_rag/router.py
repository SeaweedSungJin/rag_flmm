"""Role facade for the router component (FrozenLlavaSAM)."""
from __future__ import annotations

from .vendor_paths import ensure_vendor_paths

ensure_vendor_paths()

try:
    from integration import RoutableFLMMPipeline  # type: ignore
except Exception:
    from rag_flmm.integration.wikipedia_bridge import RoutableFLMMPipeline  # type: ignore

__all__ = ["RoutableFLMMPipeline"]

