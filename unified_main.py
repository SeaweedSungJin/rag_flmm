#!/usr/bin/env python3
"""Single-file runner for the unified FrozenLlavaSAM + Wikipedia RAG pipeline.

Usage (no extra setup besides filling data paths if needed):

  python unified_main.py "질문" /path/to/image.jpg

If ``config/unified.yaml`` exists you can still use the richer CLI:

  python unified_main.py --config config/unified.yaml "질문" /path/to/image.jpg

Environment overrides (optional):
  FLMM_DEVICE, FLMM_CHECKPOINT, WIKI_BASE_PATH, WIKI_IMAGE_PATH, WIKI_KB_JSON_NAME,
  WIKI_ENABLE_NLI, WIKI_ENABLE_VLM, GENERATE_ANSWER, MAX_NEW_TOKENS
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
# Ensure our package and vendored projects are importable
for p in (SRC_DIR, ROOT / "rag_flmm", ROOT / "wikipedia"):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from flmm_wiki_rag.config import (
    UnifiedConfig,
    FLMMConfig,
    WikipediaConfig,
    RuntimeConfig,
)
from flmm_wiki_rag.bridge import UnifiedRAGPipeline


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def build_default_config() -> UnifiedConfig:
    """Construct a UnifiedConfig using sensible project-relative defaults.

    - FLMM config points at the vendored mmengine config under ``rag_flmm``.
    - Wikipedia config points at ``wikipedia/config.yaml``.
    - Optional paths can be overridden via environment variables.
    """

    # FLMM (router) block
    flmm_cfg_path = ROOT / "rag_flmm" / "configs" / "llava" / (
        "frozen_llava_1_5_vicuna_7b_unet_sam_l_refcoco_png.py"
    )
    if not flmm_cfg_path.exists():
        raise FileNotFoundError(
            f"FrozenLlavaSAM config not found: {flmm_cfg_path}. "
            f"Please set flmm.config_path or check the vendored tree."
        )
    flmm_checkpoint = os.getenv("FLMM_CHECKPOINT", None)
    flmm_device = os.getenv("FLMM_DEVICE", "cuda")
    flmm = FLMMConfig(
        config_path=flmm_cfg_path,
        checkpoint_path=(Path(flmm_checkpoint) if flmm_checkpoint else None),
        device=flmm_device,
        rag_threshold=None,
        router_kwargs={},
    )

    # Wikipedia block
    wiki_root = ROOT / "wikipedia"
    wiki_yaml = wiki_root / "config.yaml"
    if not wiki_yaml.exists():
        raise FileNotFoundError(
            f"Wikipedia config not found: {wiki_yaml}. "
            f"Please ensure the vendored 'wikipedia' tree exists."
        )
    overrides: dict = {}
    if os.getenv("WIKI_BASE_PATH"):
        overrides["base_path"] = os.getenv("WIKI_BASE_PATH")
    if os.getenv("WIKI_IMAGE_PATH"):
        overrides["image_path"] = os.getenv("WIKI_IMAGE_PATH")
    if os.getenv("WIKI_KB_JSON_NAME"):
        overrides["kb_json_name"] = os.getenv("WIKI_KB_JSON_NAME")

    wiki_enable_nli = _as_bool(os.getenv("WIKI_ENABLE_NLI"), True)
    wiki_enable_vlm = _as_bool(os.getenv("WIKI_ENABLE_VLM"), True)
    wiki = WikipediaConfig(
        root=wiki_root,
        config_path=wiki_yaml,
        enable_nli=wiki_enable_nli,
        enable_vlm=wiki_enable_vlm,
        overrides=overrides,
    )

    # Runtime block
    rt_generate = _as_bool(os.getenv("GENERATE_ANSWER"), True)
    rt_tokens = int(os.getenv("MAX_NEW_TOKENS", "64"))
    runtime = RuntimeConfig(
        generate_answer=rt_generate, max_new_tokens=rt_tokens, pretty_print=True, json_output=None
    )

    return UnifiedConfig(flmm=flmm, wikipedia=wiki, runtime=runtime)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-file unified runner: FLMM router + Wikipedia RAG",
    )
    # Make positionals optional; we will fall back to wikipedia/config.yaml
    # (text_query, image_path) when not provided.
    p.add_argument("question", nargs="?", help="User question")
    p.add_argument("image", nargs="?", help="Path to the query image")
    p.add_argument(
        "--config",
        help="Optional unified YAML config (bypasses defaults)",
    )
    p.add_argument("--rag-threshold", type=float, default=None)
    p.add_argument("--no-generate", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.config:
        cfg = UnifiedConfig.from_yaml(args.config)
    else:
        cfg = build_default_config()

    if args.rag_threshold is not None:
        cfg.flmm.rag_threshold = args.rag_threshold
    if args.no_generate:
        cfg.runtime.generate_answer = False
    if args.max_new_tokens is not None:
        cfg.runtime.max_new_tokens = args.max_new_tokens

    # Resolve question/image via CLI, unified overrides, or wikipedia config.yaml fallback
    q = args.question
    img = args.image
    if (q is None or img is None) and cfg.wikipedia and cfg.wikipedia.overrides:
        q = q or cfg.wikipedia.overrides.get("text_query")
        img = img or cfg.wikipedia.overrides.get("image_path")
    if (q is None or img is None) and cfg.wikipedia and cfg.wikipedia.config_path:
        try:
            with Path(cfg.wikipedia.config_path).open("r", encoding="utf-8") as f:
                wcfg = yaml.safe_load(f) or {}
            q = q or wcfg.get("text_query")
            img = img or wcfg.get("image_path")
        except Exception:
            pass
    if not q or not img:
        raise SystemExit(
            "question/image not provided and not found in wikipedia config.yaml. "
            "Pass them on the command line or set in wikipedia/config.yaml."
        )

    pipeline = UnifiedRAGPipeline(cfg)
    pipeline.run_and_dump(
        question=str(q),
        image_path=Path(str(img)),
        rag_threshold=args.rag_threshold,
        generate_answer=not args.no_generate,
        max_new_tokens=args.max_new_tokens,
        json_path=cfg.runtime.json_output,
        pretty_print=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
