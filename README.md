# Compress-Anomaly Phase 1 Harness

Phase 1 research harness: does compression preserve anomaly signal?

## Quickstart

Always run from the repo root (`/home/shreyas/projects/compress-anamoly/`).
After `uv sync`, the package imports as `harness.*`.

```bash
uv sync
uv run python -c "import sys; print(sys.version)"
uv run pytest -q
```

## Layout

- `harness/datasets/` — synthetic generator + UCR loader (skeleton)
- `harness/codecs/` — compression codecs R0/R3/R4 (skeleton)
- `harness/detectors/` — detectors PCA/IF/D4 (skeleton)
- `harness/metrics/` — frozen evaluator, alignment, VUS-PR (skeleton)
- `harness/results/` — run outputs (gitignored contents)
- `tests/` — pytest suite
- `docs/` — ARCHITECTURE.md, METRICS.md, EVAL.md
- `eval/fixtures/` — eval fixtures (skeleton)
- `data/cache/` — UCR download cache (gitignored)
- `dataset-freeze.csv` — pinned dataset freeze record (header only until Task 4)
