"""Ensure vendored repositories are importable.

The unified project vendors two upstream trees under the project root:

- ``rag_flmm`` – FrozenLlavaSAM router and training utilities
- ``wikipedia`` – Wikipedia RAG retrieval pipeline

This helper prepends those directories to ``sys.path`` so their modules can be
imported without requiring separate editable installs.
"""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_vendor_paths() -> None:
    here = Path(__file__).resolve()
    # This file lives at: <repo>/flmm_wiki_rag/src/flmm_wiki_rag/vendor_paths.py
    # Project root is two levels up from 'src/flmm_wiki_rag': parents[2] == .../flmm_wiki_rag
    project_root = here.parents[2]
    vendor_dirs = [project_root / "rag_flmm", project_root / "wikipedia"]
    for d in vendor_dirs:
        ds = str(d)
        if d.exists() and ds not in sys.path:
            sys.path.insert(0, ds)


__all__ = ["ensure_vendor_paths"]
