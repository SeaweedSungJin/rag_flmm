"""High-level wrapper combining FrozenLlavaSAM routing with Wikipedia RAG."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mmengine.config import Config as MMConfig
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER

from .config import UnifiedConfig
from .vendor_paths import ensure_vendor_paths

# Ensure vendored repos are importable
ensure_vendor_paths()

# Import router/retriever bridge from vendored integration
try:
    from integration import (  # type: ignore  # noqa: E402
        RAGRetrievalResult,
        RoutableFLMMPipeline,
        WikipediaRAGBridge,
    )
except Exception:
    from rag_flmm.integration.wikipedia_bridge import (  # type: ignore  # noqa: E402
        RAGRetrievalResult,
        RoutableFLMMPipeline,
        WikipediaRAGBridge,
    )


@dataclass
class UnifiedResult:
    """Structured output returned by :class:`UnifiedRAGPipeline`."""

    question: str
    image_path: str
    router_prob: float
    router_threshold: float
    use_rag: bool
    retrieval: RAGRetrievalResult

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        payload: Dict[str, Any] = {
            "question": self.question,
            "image_path": self.image_path,
            "router_prob": self.router_prob,
            "router_threshold": self.router_threshold,
            "use_rag": self.use_rag,
            "retrieval": {
                "use_retrieval": self.retrieval.use_retrieval,
                "search_elapsed": self.retrieval.search_elapsed,
                "nli_elapsed": self.retrieval.nli_elapsed,
                "sections": self.retrieval.sections,
                "section_metadata": self.retrieval.section_metadata,
                "image_results": self.retrieval.image_results,
                "answer": self.retrieval.answer,
            },
        }
        return payload


class UnifiedRAGPipeline:
    """End-to-end wrapper exposing a simple ``run`` method."""

    def __init__(self, config: UnifiedConfig) -> None:
        self.config = config
        self._mm_cfg: Optional[MMConfig] = None
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._prompt_template: Dict[str, Any] | None = None
        self._image_token: str | None = None
        self._rag_token: str | None = None
        self._rag_bridge: Optional[WikipediaRAGBridge] = None
        self._pipeline: Optional[RoutableFLMMPipeline] = None

        self._load_components()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_components(self) -> None:
        self._load_flmm()
        self._load_bridge()
        self._pipeline = RoutableFLMMPipeline(
            model=self._model,
            rag_bridge=self._rag_bridge,
            tokenizer=self._tokenizer,
            image_processor=self._image_processor,
            prompt_template=self._prompt_template,
            image_token=self._image_token or "<image>",
            rag_token=self._rag_token or "[RAG]",
        )

    def _load_flmm(self) -> None:
        cfg_path = self.config.flmm.config_path
        if not cfg_path.exists():
            raise FileNotFoundError(f"FrozenLlavaSAM config not found: {cfg_path}")

        self._mm_cfg = MMConfig.fromfile(str(cfg_path))
        device = self.config.flmm.device
        model = BUILDER.build(self._mm_cfg.model)
        model = model.to(device)
        checkpoint = self.config.flmm.checkpoint_path
        if checkpoint is not None:
            if not checkpoint.exists():
                raise FileNotFoundError(f"FrozenLlavaSAM checkpoint not found: {checkpoint}")
            state = guess_load_checkpoint(str(checkpoint))
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[flmm_wiki_rag] Missing keys while loading checkpoint ({len(missing)}): {missing[:8]}")
            if unexpected:
                print(
                    f"[flmm_wiki_rag] Unexpected keys while loading checkpoint ({len(unexpected)}):"
                    f" {unexpected[:8]}"
                )
        model.eval()
        self._model = model

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tok_cfg = getattr(self._mm_cfg, "tokenizer", None)
            if tok_cfg is None:
                raise RuntimeError("Tokenizer config missing from FrozenLlavaSAM config")
            tokenizer = BUILDER.build(tok_cfg)
            # Try to find an underlying HF model to resize embeddings on
            base = (
                getattr(model, "llm", None)
                or getattr(model, "llava", None)
                or getattr(model, "language_model", None)
                or getattr(model, "model", None)
                or None
            )
            if base is not None and hasattr(base, "resize_token_embeddings"):
                try:
                    base.resize_token_embeddings(len(tokenizer))
                except Exception:
                    pass
        self._tokenizer = tokenizer

        img_proc_cfg = getattr(self._mm_cfg, "image_processor", None)
        if img_proc_cfg is None:
            raise RuntimeError("Image processor config missing from FrozenLlavaSAM config")
        self._image_processor = BUILDER.build(img_proc_cfg)

        prompt_template = getattr(self._mm_cfg, "prompt_template", None)
        if prompt_template is None or "INSTRUCTION" not in prompt_template:
            raise RuntimeError("prompt_template with INSTRUCTION key is required")
        self._prompt_template = prompt_template

        self._image_token = getattr(self._mm_cfg, "image_token", "<image>")
        self._rag_token = getattr(self._model, "rag_token", "[RAG]")

    def _load_bridge(self) -> None:
        wiki_root = self.config.wikipedia.root
        if not wiki_root.exists():
            raise FileNotFoundError(f"Wikipedia project root not found: {wiki_root}")
        cfg_path = self.config.wikipedia.config_path
        if not cfg_path.exists():
            raise FileNotFoundError(f"Wikipedia config not found: {cfg_path}")

        self._rag_bridge = WikipediaRAGBridge(
            wikipedia_root=str(wiki_root),
            config_path=str(cfg_path),
            config_overrides=self.config.wikipedia.overrides,
            enable_nli=self.config.wikipedia.enable_nli,
            enable_vlm=self.config.wikipedia.enable_vlm,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        question: str,
        image_path: str | Path,
        *,
        rag_threshold: Optional[float] = None,
        force_use_rag: Optional[bool] = None,
        generate_answer: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
    ) -> UnifiedResult:
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialised")

        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        runtime = self.config.runtime
        gen_answer = runtime.generate_answer if generate_answer is None else generate_answer
        max_tokens = runtime.max_new_tokens if max_new_tokens is None else max_new_tokens
        rag_thresh = (
            self.config.flmm.rag_threshold if rag_threshold is None else rag_threshold
        )

        route_kwargs = dict(self.config.flmm.router_kwargs)
        result = self._pipeline.route(
            question=question,
            image_path=img_path,
            rag_threshold=rag_thresh,
            force_use_rag=force_use_rag,
            generate_answer=gen_answer,
            max_new_tokens=max_tokens,
            **route_kwargs,
        )

        return UnifiedResult(
            question=question,
            image_path=str(img_path),
            router_prob=result["router_prob"],
            router_threshold=result["router_threshold"],
            use_rag=result["use_rag"],
            retrieval=result["retrieval"],
        )

    def run_to_json(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute ``run`` and return a JSON-compatible dict."""
        return self.run(*args, **kwargs).to_json()

    def run_and_dump(
        self,
        question: str,
        image_path: str | Path,
        *,
        rag_threshold: Optional[float] = None,
        force_use_rag: Optional[bool] = None,
        generate_answer: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        json_path: Optional[Path] = None,
        pretty_print: bool | None = None,
    ) -> UnifiedResult:
        """Run the pipeline and optionally write the JSON summary to disk."""
        result = self.run(
            question,
            image_path,
            rag_threshold=rag_threshold,
            force_use_rag=force_use_rag,
            generate_answer=generate_answer,
            max_new_tokens=max_new_tokens,
        )
        payload = result.to_json()

        if json_path is not None:
            json_path = Path(json_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        do_pretty = self.config.runtime.pretty_print if pretty_print is None else pretty_print
        if do_pretty:
            self._print_summary(result)

        return result

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(result: UnifiedResult) -> None:
        retrieval = result.retrieval
        print("=== Router Decision ===")
        print(
            f"P(use_rag) = {result.router_prob:.4f} "
            f"(threshold {result.router_threshold:.4f})"
        )
        print(f"Using retrieval: {result.use_rag}")
        print()

        if retrieval.sections:
            print("=== Retrieved Context ===")
            for idx, (meta, text) in enumerate(
                zip(retrieval.section_metadata, retrieval.sections), 1
            ):
                title = meta.get("source_title") or meta.get("doc_title") or "unknown"
                section_title = meta.get("section_title") or meta.get("section_id")
                print(f"[{idx}] {title} :: {section_title}")
                snippet = text.strip().replace("\n", " ")
                print(snippet[:400])
                if len(snippet) > 400:
                    print("...")
                print()
        else:
            print("No sections were selected for this query.")
            print()

        if retrieval.answer is not None:
            print("=== Generated Answer ===")
            print(retrieval.answer)
            print()

        print("=== JSON Summary ===")
        print(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
