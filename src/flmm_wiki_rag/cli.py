"""Command-line interface for the unified FLMM + Wikipedia RAG pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .bridge import UnifiedRAGPipeline
from .config import UnifiedConfig


def _parse_key_value(items: list[str]) -> Dict[str, Any]:
    def convert(value: str) -> Any:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    result: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"Invalid key in '{item}'")
        result[key] = convert(value.strip())
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FrozenLlavaSAM routing with the Wikipedia RAG backend",
    )
    # Positional arguments are optional; we fall back to values from
    # wikipedia/config.yaml (text_query, image_path) if not supplied.
    parser.add_argument("question", nargs="?", help="User question to answer")
    parser.add_argument("image", nargs="?", help="Path to the query image")
    parser.add_argument(
        "--config",
        default="config/unified.yaml",
        help="Path to the unified YAML configuration",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to dump the JSON summary",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Disable pretty console output (JSON is still printed)",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip VLM answer generation even if enabled in config",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override the VLM max_new_tokens value",
    )
    parser.add_argument(
        "--rag-threshold",
        type=float,
        help="Override the router threshold",
    )
    parser.add_argument(
        "--force-use-rag",
        action="store_true",
        help="Force retrieval regardless of router output",
    )
    parser.add_argument(
        "--force-no-rag",
        action="store_true",
        help="Skip retrieval regardless of router output",
    )
    parser.add_argument(
        "--device",
        help="Override the device for the FrozenLlavaSAM model",
    )
    parser.add_argument(
        "--wiki-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a key in the Wikipedia config (applied post-YAML)",
    )
    parser.add_argument(
        "--router-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra keyword arguments for RoutableFLMMPipeline.route",
    )
    # Optional flags mirroring the positionals for convenience
    parser.add_argument("--question", dest="question_text", help="Question (overrides config)")
    parser.add_argument("--image", dest="image_path", help="Image path (overrides config)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        parser.error(f"Config file not found: {cfg_path}")

    cfg = UnifiedConfig.from_yaml(cfg_path)

    if args.device:
        cfg.flmm.device = args.device
    if args.json_output:
        cfg.runtime.json_output = Path(args.json_output)
    if args.no_pretty:
        cfg.runtime.pretty_print = False
    if args.no_generate:
        cfg.runtime.generate_answer = False
    if args.max_new_tokens is not None:
        cfg.runtime.max_new_tokens = args.max_new_tokens
    if args.rag_threshold is not None:
        cfg.flmm.rag_threshold = args.rag_threshold
    if args.wiki_override:
        cfg.wikipedia.overrides.update(_parse_key_value(args.wiki_override))
    if args.router_arg:
        cfg.flmm.router_kwargs.update(_parse_key_value(args.router_arg))

    force_use_rag = None
    if args.force_use_rag and args.force_no_rag:
        parser.error("--force-use-rag and --force-no-rag are mutually exclusive")
    if args.force_use_rag:
        force_use_rag = True
    elif args.force_no_rag:
        force_use_rag = False

    # Resolve question/image from CLI or fallback to unified overrides, then wikipedia config.yaml
    q = args.question or getattr(args, "question_text", None)
    img = args.image or getattr(args, "image_path", None)
    # unified.yaml overrides take precedence if provided
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
        parser.error("question/image not provided and not found in wikipedia config.yaml")

    pipeline = UnifiedRAGPipeline(cfg)
    result = pipeline.run_and_dump(
        question=str(q),
        image_path=Path(str(img)),
        rag_threshold=args.rag_threshold,
        force_use_rag=force_use_rag,
        generate_answer=not args.no_generate,
        max_new_tokens=args.max_new_tokens,
        json_path=cfg.runtime.json_output,
        pretty_print=not args.no_pretty,
    )

    if not cfg.runtime.pretty_print:
        print(json.dumps(result.to_json(), ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
