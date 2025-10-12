"""Role facade for the retrieval component (Wikipedia RAG)."""
from __future__ import annotations

from .vendor_paths import ensure_vendor_paths

ensure_vendor_paths()

try:
    from integration import WikipediaRAGBridge, RAGRetrievalResult  # type: ignore
except Exception:
    from rag_flmm.integration.wikipedia_bridge import (  # type: ignore
        WikipediaRAGBridge,
        RAGRetrievalResult,
    )

try:
    # Direct access to the underlying search pipeline & config
    from src.pipeline import search_rag_pipeline  # type: ignore
    from src.config import Config as RetrievalConfig  # type: ignore
except Exception:
    # In case import paths differ, leave these unexported
    search_rag_pipeline = None  # type: ignore
    RetrievalConfig = None  # type: ignore

__all__ = [
    "WikipediaRAGBridge",
    "RAGRetrievalResult",
    "search_rag_pipeline",
    "RetrievalConfig",
]

