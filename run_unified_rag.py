#!/usr/bin/env python3
"""Unified runner for FrozenLlavaSAM routing + Wikipedia RAG.

Behavior:
- If a unified YAML exists (default: config/unified.yaml or via --config), use
  the richer CLI which supports overrides and JSON output.
- If no config is provided and the default is missing, fall back to the
  zero-config runner which derives defaults from the vendored trees.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Detect whether a --config flag was provided or a default exists.
    has_flag = any(arg == "--config" or arg.startswith("--config=") for arg in argv)
    default_cfg_primary = ROOT / "config" / "config.yaml"
    default_cfg_fallback = ROOT / "config" / "unified.yaml"
    default_cfg = default_cfg_primary if default_cfg_primary.exists() else default_cfg_fallback
    use_cli = has_flag or default_cfg.exists()

    if use_cli:
        from flmm_wiki_rag.cli import main as cli_main

        # If user passed --config without a value, append default path to avoid
        # argparse complaining; otherwise leave argv as-is.
        if "--config" in argv:
            idx = argv.index("--config")
            if idx == len(argv) - 1 or argv[idx + 1].startswith("-"):
                argv[idx:idx + 1] = ["--config", str(default_cfg)]
        cli_main(argv)
    else:
        # Zero-config path: build defaults from vendored repos.
        from unified_main import main as zero_main  # type: ignore

        zero_main(argv)


if __name__ == "__main__":  # pragma: no cover
    main()
