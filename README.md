# FLMM + Wikipedia RAG

Unified project that bundles the FrozenLlavaSAM router and the Wikipedia RAG
retrieval stack into a single, self-contained folder.  A
question/image pair is first routed through FrozenLlavaSAM to decide whether
external retrieval is needed.  When the router fires, the Wikipedia pipeline
runs FAISS image search, text segmentation, optional NLI pruning, and
(Optionally) vision-language answer generation.

## Repository layout

```
flmm_wiki_rag/
├── run_unified_rag.py       # Executable entrypoint
├── config/
│   └── unified.example.yaml
├── rag_flmm/                # Vendored FrozenLlavaSAM project
├── src/flmm_wiki_rag/
│   ├── __init__.py
│   ├── __main__.py
│   ├── bridge.py
│   ├── cli.py
│   └── config.py
└── wikipedia/               # Vendored Wikipedia RAG project
```

Everything needed to run lives inside this directory—no external checkout is
required.

## Quick start

1. Copy `config/unified.example.yaml` to `config/unified.yaml` and update the
   relative paths if needed (for example set `flmm.config_path` to
   `../rag_flmm/...` and tweak the Wikipedia dataset locations).
2. Install requirements in your environment (`pip install -e .` from the
   `flmm_wiki_rag` directory).
3. Execute the standalone script with your question and image:

```bash
python run_unified_rag.py \
    --config config/unified.yaml \
    "What is this monument called?" \
    /path/to/query-image.jpg
```

The script prints router diagnostics, retrieved evidence, and the optional VLM
answer.  A JSON summary is also written to stdout (and optionally saved to
file).

## Configuration overview

The unified YAML file has three sections:

- `flmm` – path to the FrozenLlavaSAM mmengine config, optional checkpoint,
  target device, and router threshold override.
- `wikipedia` – root directory of the Wikipedia project, config file, plus
  optional key/value overrides applied after loading the YAML.
- `runtime` – toggles for VLM generation, maximum new tokens, pretty-printing,
  and JSON export path.

See `config/unified.example.yaml` for the full list of keys and inline
comments.

## Programmatic usage

```python
from flmm_wiki_rag.bridge import UnifiedRAGPipeline, UnifiedConfig

cfg = UnifiedConfig.from_yaml("config/unified.yaml")
pipeline = UnifiedRAGPipeline(cfg)
result = pipeline.run(
    question="Who commissioned this abbey?",
    image="datasets/real_images/Downside_Abbey.jpg",
)
print(result.answer)
```

The result object contains router scores, retrieved sections, metadata, and the
optional generated answer.  All heavy models remain cached across subsequent
calls.

## Next steps

- Integrate into evaluation scripts or services by reusing
  `UnifiedRAGPipeline`.
- Extend the YAML schema with override knobs (e.g. switch rerankers, adjust
  batch sizes).
- Add tests or batching wrappers if you plan to serve the pipeline in
  production.
