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
    p.add_argument("--eval-em", action="store_true", help="Compute Exact Match (EM) against CSV answer (default on)")
    p.add_argument("--eval-bem", action="store_true", help="Compute BEM score via TF-Hub model (network + TF required; default on)")
    p.add_argument("--no-eval-em", action="store_true", help="Disable EM evaluation")
    p.add_argument("--no-eval-bem", action="store_true", help="Disable BEM evaluation")
    return p


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    # Defaults: enable EM/BEM unless explicitly disabled
    parser.set_defaults(eval_em=True, eval_bem=True)
    args = parser.parse_args(argv)
    if args.no_eval_em:
        args.eval_em = False
    if args.no_eval_bem:
        args.eval_bem = False

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
    oom_warned = False

    # Optional evaluators (lazy import to avoid heavy deps if not needed)
    eval_funcs: Dict[str, Any] = {}
    if args.eval_em or args.eval_bem:
        try:
            from src.eval import preprocess_answer as _preprocess_answer  # type: ignore
            eval_funcs["preprocess_answer"] = _preprocess_answer
        except Exception:
            # Fallback local normaliser
            import re

            def _simple_norm(s: str) -> str:
                s = (s or "").lower().strip()
                s = re.sub(r"\s+", " ", s)
                return s

            eval_funcs["preprocess_answer"] = _simple_norm
    if args.eval_bem:
        try:
            from src.eval import evaluate_example as _evaluate_example  # type: ignore

            eval_funcs["evaluate_example"] = _evaluate_example
        except Exception as e:
            print(f"[WARN] BEM evaluator unavailable: {e}. --eval-bem will be ignored.")
            args.eval_bem = False

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
            # Optional: EM/BEM scoring against ground truth answer
            em_score: Optional[float] = None
            bem_score: Optional[float] = None
            correct: Optional[float] = None
            if (args.eval_em or args.eval_bem) and isinstance(sample.answer, str):
                ref = sample.answer
                pred = payload.get("retrieval", {}).get("answer") or ""
                norm = eval_funcs.get("preprocess_answer")
                if args.eval_em and norm:
                    try:
                        ref_norm = norm(ref)
                        pred_norm = norm(pred)
                        em_score = 1.0 if ref_norm == pred_norm and ref_norm != "" else 0.0
                    except Exception:
                        em_score = 0.0
                if args.eval_bem and eval_funcs.get("evaluate_example"):
                    try:
                        meta_qtype = sample.metadata.get("question_type") if isinstance(sample.metadata, dict) else None
                        qtype = str(meta_qtype) if meta_qtype else ("multi_answer" if ("&&" in ref) else "automatic")
                        bem_score = float(eval_funcs["evaluate_example"](sample.question, [ref], pred, qtype))
                    except Exception as e:
                        print(f"[WARN] BEM failed on row {sample.row_idx}: {e}")
                        bem_score = None
                # Final correctness: EM win OR (EM fail and BEM >= 0.5)
                try:
                    if em_score is not None and em_score >= 1.0:
                        correct = 1.0
                    elif bem_score is not None:
                        correct = 1.0 if bem_score >= 0.5 else 0.0
                    elif em_score is not None:
                        correct = em_score
                except Exception:
                    correct = None
            if em_score is not None:
                payload["em"] = float(em_score)
            if bem_score is not None:
                payload["bem"] = float(bem_score)
            if correct is not None:
                payload["correct"] = float(correct)

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            processed += 1
        except Exception as e:  # noqa: BLE001
            err_text = str(e)
            failed.append({"row": global_idx, "error": err_text})
            if not oom_warned and "out of memory" in err_text.lower():
                print(f"[WARN] CUDA OOM encountered on row {global_idx}. Consider adjusting GPU assignments or batch sizes.")
                oom_warned = True

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
                    "em": data.get("em"),
                    "bem": data.get("bem"),
                    "correct": data.get("correct"),
                }
            )
        except Exception:
            continue
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_root / "manifest.csv", index=False)
        # Compute aggregates and update summary
        em_vals = [r.get("em") for r in rows if r.get("em") is not None]
        bem_vals = [r.get("bem") for r in rows if r.get("bem") is not None]
        correct_vals = [r.get("correct") for r in rows if r.get("correct") is not None]
        if em_vals:
            summary["em_avg"] = float(sum(em_vals) / len(em_vals))
        if bem_vals:
            summary["bem_avg"] = float(sum(bem_vals) / len(bem_vals))
        if correct_vals:
            correct_count = sum(1 for v in correct_vals if v >= 0.5)
            evaluated_count = len(correct_vals)
            incorrect_count = evaluated_count - correct_count
            summary["evaluated_count"] = int(evaluated_count)
            summary["correct_count"] = int(correct_count)
            summary["incorrect_count"] = int(incorrect_count)
            summary["accuracy"] = correct_count / evaluated_count
            if summary.get("total_rows"):
                summary["accuracy_over_total"] = correct_count / summary["total_rows"]
        # counts for use_rag
        use_true = sum(1 for r in rows if r.get("use_rag") is True)
        use_false = sum(1 for r in rows if r.get("use_rag") is False)
        summary["use_rag_true_count"] = int(use_true)
        summary["use_rag_false_count"] = int(use_false)
        # Rewrite summary with aggregates
        with (out_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
