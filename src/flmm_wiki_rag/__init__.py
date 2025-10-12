"""Unified FrozenLlavaSAM + Wikipedia RAG pipeline.

This package ensures vendored projects (``rag_flmm`` and ``wikipedia``) are on
``sys.path`` and exposes stable, role-oriented APIs.
"""

from . import vendor_paths as _vendor_paths  # noqa: F401
_vendor_paths.ensure_vendor_paths()

from .config import UnifiedConfig, FLMMConfig, WikipediaConfig, RuntimeConfig
from .bridge import UnifiedRAGPipeline, UnifiedResult

__all__ = [
    "UnifiedConfig",
    "FLMMConfig",
    "WikipediaConfig",
    "RuntimeConfig",
    "UnifiedRAGPipeline",
    "UnifiedResult",
]
