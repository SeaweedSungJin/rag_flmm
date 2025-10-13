#!/usr/bin/env python3
"""Batch runner for the unified FLMM + Wikipedia RAG pipeline.

Iterates over a VQA-style CSV (as used in the wikipedia project), runs the
router → RAG pipeline for each row, and writes per-sample JSON plus a summary
report. Supports resume and simple sharding for multi-GPU/process runs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

# Ensure our package and vendored projects are importable
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from flmm_wiki_rag.config import UnifiedConfig
from flmm_wiki_rag.bridge import UnifiedRAGPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="Unified YAML config file (defaults to config/config.yaml or config/unified.yaml)")
    p.add_argument("--output", default=str(ROOT / "runs"), help="Output directory root")
    p.add_argument("--start", type=int, default=0, help="Start row offset")
    p.add_argument("--end", type=int, default=None, help="End row offset (exclusive)")
    p.add_argument("--shard", type=int, default=0, help="Current shard index (0-based)")
    p.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    p.add_argument("--resume", action="store_true", help="Skip rows that already produced a result JSON")
    p.add_argument("--max-new-tokens", type=int, default=None, help="Override VLM max_new_tokens")
    p.add_argument("--rag-threshold", type=float, default=None, help="Override router threshold")
    return p


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    # Resolve config file with fallbacks
    if args.config is None:
        candidates = [ROOT / "config" / "config.yaml", ROOT / "config" / "unified.yaml", ROOT / "config" / "unified.example.yaml"]
        cfg_path = next((str(p) for p in candidates if p.exists()), None)
        if cfg_path is None:
            raise FileNotFoundError("No config found. Create config/config.yaml or config/unified.yaml (you can copy from config/unified.example.yaml).")
    else:
        cfg_path = args.config
    cfg = UnifiedConfig.from_yaml(cfg_path)
    out_root = Path(args.output)
    _ensure_dir(out_root)
    out_samples = out_root / "samples"
    _ensure_dir(out_samples)

    # Snapshot config for reproducibility
    with (out_root / "config_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

    # Build the unified pipeline (loads router + RAG stack once)
    pipeline = UnifiedRAGPipeline(cfg)

    # Load dataset CSV using the wikipedia dataloader schema
    # Prefer unified overrides but fall back to the wikipedia config
    csv_path = cfg.wikipedia.overrides.get("dataset_csv") if cfg.wikipedia.overrides else None
    if not csv_path:
        # Read from the wikipedia YAML directly
        from importlib import import_module

        # Add wikipedia/src to path via bridge; already handled inside UnifiedRAGPipeline
        from src.config import Config as WikiConfig  # type: ignore

        wcfg = WikiConfig.from_yaml(str(cfg.wikipedia.config_path))
        csv_path = wcfg.dataset_csv
        id2name_paths = wcfg.id2name_paths
        image_root = wcfg.dataset_image_root
        google_root = wcfg.dataset_google_root
    else:
        id2name_paths = cfg.wikipedia.overrides.get("id2name_paths") if cfg.wikipedia.overrides else None
        image_root = cfg.wikipedia.overrides.get("dataset_image_root") if cfg.wikipedia.overrides else None
        google_root = cfg.wikipedia.overrides.get("dataset_google_root") if cfg.wikipedia.overrides else None

    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError(f"dataset_csv not found: {csv_path}")

    # Use the official dataloader to parse rows/resolve image paths
    from src.dataloader import VQADataset  # type: ignore

    ds = VQADataset(
        csv_path=str(csv_path),
        id2name_paths=id2name_paths,
        image_root=image_root,
        googlelandmark_root=google_root,
        start=args.start,
        end=args.end,
    )

    processed = 0
    failed: list[dict[str, Any]] = []

    for i, sample in enumerate(tqdm(ds, total=len(ds), desc="Batch RAG")):
        # Sharding support
        global_idx = sample.row_idx
        if args.num_shards > 1:
            if (global_idx % args.num_shards) != args.shard:
                continue

        # Choose the first resolved image path per sample for now
        if not sample.image_paths:
            failed.append({"row": global_idx, "error": "no_image_path"})
            continue
        img_path = sample.image_paths[0]

        # Resume support: if JSON exists, skip
        out_path = out_samples / f"row{global_idx}.json"
        if args.resume and out_path.exists():
            continue

        try:
            result = pipeline.run(
                question=sample.question,
                image_path=img_path,
                rag_threshold=args.rag_threshold,
                generate_answer=cfg.runtime.generate_answer,
                max_new_tokens=args.max_new_tokens or cfg.runtime.max_new_tokens,
            )
            # Console hint per row for router decision
            try:
                print(
                    f"[row {sample.row_idx}] router_prob={result.router_prob:.3f} "
                    f"threshold={result.router_threshold:.2f} use_rag={result.use_rag}"
                )
            except Exception:
                pass
            # Persist sample result + minimal metadata
            payload = result.to_json()
            payload["row_idx"] = int(sample.row_idx)
            payload["metadata"] = sample.metadata
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            processed += 1
        except Exception as e:  # noqa: BLE001
            failed.append({"row": global_idx, "error": str(e)})

    # Write summary
    summary = {
        "config": cfg.to_dict(),
        "output": str(out_root),
        "total_rows": int(len(ds)),
        "processed": int(processed),
        "failed": failed,
        "start": args.start,
        "end": args.end,
        "shard": args.shard,
        "num_shards": args.num_shards,
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Optional: write a CSV manifest of successes
    rows = []
    for path in out_samples.glob("row*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append(
                {
                    "row_idx": data.get("row_idx"),
                    "use_rag": data.get("use_rag"),
                    "router_prob": data.get("router_prob"),
                    "answer": data.get("retrieval", {}).get("answer"),
                }
            )
        except Exception:
            continue
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_root / "manifest.csv", index=False)


if __name__ == "__main__":  # pragma: no cover
    main()
