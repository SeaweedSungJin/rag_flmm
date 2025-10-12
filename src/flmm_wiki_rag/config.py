"""Dataclasses for the unified FLMM + Wikipedia RAG configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class FLMMConfig:
    """Configuration for loading the FrozenLlavaSAM router model."""

    config_path: Path
    checkpoint_path: Optional[Path] = None
    device: str = "cuda"
    rag_threshold: Optional[float] = None
    router_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_path: Path) -> "FLMMConfig":
        cfg_path = base_path / data["config_path"] if not Path(data["config_path"]).is_absolute() else Path(data["config_path"])
        ckpt_value = data.get("checkpoint_path")
        ckpt_path = None
        if ckpt_value:
            ckpt = Path(ckpt_value)
            ckpt_path = ckpt if ckpt.is_absolute() else base_path / ckpt
        return cls(
            config_path=cfg_path,
            checkpoint_path=ckpt_path,
            device=str(data.get("device", "cuda")),
            rag_threshold=data.get("rag_threshold"),
            router_kwargs=dict(data.get("router_kwargs", {})),
        )


@dataclass
class WikipediaConfig:
    """Configuration block for the Wikipedia RAG pipeline."""

    root: Path
    config_path: Path
    enable_nli: bool = True
    enable_vlm: bool = True
    overrides: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_path: Path) -> "WikipediaConfig":
        root_value = Path(data["root"])
        root = root_value if root_value.is_absolute() else (base_path / root_value)
        cfg_value = Path(data.get("config_path", "config.yaml"))
        cfg_path = cfg_value if cfg_value.is_absolute() else (root / cfg_value)
        return cls(
            root=root,
            config_path=cfg_path,
            enable_nli=bool(data.get("enable_nli", True)),
            enable_vlm=bool(data.get("enable_vlm", True)),
            overrides=dict(data.get("overrides", {})),
        )


@dataclass
class RuntimeConfig:
    """Runtime toggles for the unified runner."""

    generate_answer: bool = True
    max_new_tokens: int = 64
    pretty_print: bool = True
    json_output: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_path: Path) -> "RuntimeConfig":
        json_value = data.get("json_output")
        json_path = None
        if json_value:
            jp = Path(json_value)
            json_path = jp if jp.is_absolute() else (base_path / jp)
        return cls(
            generate_answer=bool(data.get("generate_answer", True)),
            max_new_tokens=int(data.get("max_new_tokens", 64)),
            pretty_print=bool(data.get("pretty_print", True)),
            json_output=json_path,
        )


@dataclass
class UnifiedConfig:
    """Top-level configuration bundling FLMM, Wikipedia, and runtime settings."""

    flmm: FLMMConfig
    wikipedia: WikipediaConfig
    runtime: RuntimeConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, base_path: Path | None = None) -> "UnifiedConfig":
        base = base_path or Path.cwd()
        flmm_block = data.get("flmm") or {}
        wiki_block = data.get("wikipedia") or {}
        runtime_block = data.get("runtime") or {}
        return cls(
            flmm=FLMMConfig.from_dict(flmm_block, base),
            wikipedia=WikipediaConfig.from_dict(wiki_block, base),
            runtime=RuntimeConfig.from_dict(runtime_block, base),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UnifiedConfig":
        yaml_path = Path(path)
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data, base_path=yaml_path.parent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flmm": {
                "config_path": str(self.flmm.config_path),
                "checkpoint_path": str(self.flmm.checkpoint_path) if self.flmm.checkpoint_path else None,
                "device": self.flmm.device,
                "rag_threshold": self.flmm.rag_threshold,
                "router_kwargs": self.flmm.router_kwargs,
            },
            "wikipedia": {
                "root": str(self.wikipedia.root),
                "config_path": str(self.wikipedia.config_path),
                "enable_nli": self.wikipedia.enable_nli,
                "enable_vlm": self.wikipedia.enable_vlm,
                "overrides": self.wikipedia.overrides,
            },
            "runtime": {
                "generate_answer": self.runtime.generate_answer,
                "max_new_tokens": self.runtime.max_new_tokens,
                "pretty_print": self.runtime.pretty_print,
                "json_output": str(self.runtime.json_output) if self.runtime.json_output else None,
            },
        }
