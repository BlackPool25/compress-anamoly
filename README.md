# Compress-Anomaly Phase 1 Harness

Phase 1 research harness: does compression preserve anomaly signal?

Five datasets (Spike, Rhythm, Drift, Chaos, pinned UCR) cross four codec
rungs (R0, Q8, Q4, R4) cross three detectors (PCA, IF, D4) under a frozen
tau rule. Per-dataset means decide; nothing is ever pooled.

## Quickstart

Always run from the repo root (`/home/shreyas/projects/compress-anamoly/`).
After `uv sync`, the package imports as `harness.*`.

```bash
uv sync
uv run pytest -q
uv run python harness/run_harness.py --smoke
uv run python harness/run_harness.py
uv run python eval/check_expected.py
```

What each step does:

1. `uv sync` — install the pinned stack (Python 3.11, numpy 1.26.x, vus 0.0.6).
2. `uv run pytest -q` — full suite, must stay green (84 passed, 1 skipped).
3. `--smoke` — 4-row fast check (seed 42, Spike only, R0+Q8); ~seconds.
4. Full run — 180 rows (5 seeds x exact matrix incl. UCR download) into
   `harness/results/baseline_run.csv`.
5. Evaluation gates — asserts non-collapse, grammar win, and quantization cliffs
   against `harness/results/baseline_run.csv` (exits 0, all 4 gates pass).
   Can also be run against fixtures (e.g. `--csv eval/fixtures/pass.csv`).


Hygiene checks:

```bash
uv run python eval/check_docstrings.py
grep -rniE 'paa.*16|pa%k|pate|affiliation|point_adjust' harness/ eval/ || true
```

The sweep must exit 0. The grep covers parked scope items; its only
acceptable hits are the PA-ban enforcer in `eval/check_expected.py` naming
the tokens it bans (no parked item has a code path).

## Layout

- `harness/datasets/` — seeded 4-morphology generator + pinned UCR loader
- `harness/codecs/` — R0 lossless, R3 Q8/Q4 quantization, R4 SAX symbolic
- `harness/detectors/` — PCA + Isolation Forest (w=32) + D4 direct-symbolic
- `harness/metrics/` — frozen-tau evaluator, trailing alignment, VUS-PR
- `harness/run_harness.py` — matrix runner (CSV + per-dataset tables)
- `harness/results/` — run outputs (untracked contents)
- `eval/check_expected.py` — bump/cliff gates + PA-ban + freeze audit
- `eval/check_docstrings.py` — docstring sweep (stdlib ast, exit 1 on gaps)
- `eval/fixtures/` — pass + four fail-mode fixtures (14-col schema)
- `tests/` — pytest suite
- `docs/` — ARCHITECTURE.md, METRICS.md, EVAL.md
- `data/cache/` — UCR download cache (gitignored)
- `dataset-freeze.csv` — pinned UCR freeze record (sha256, bytes, split rule)

## Docs

- `docs/ARCHITECTURE.md` — cell routing matrix, PLAN-FREEZE register,
  PARKED register with re-entry conditions, TRACEABILITY appendix.
- `docs/METRICS.md` — why these four metrics, gameability note, VUS rationale.
- `docs/EVAL.md` — expected phenomena, gate inequalities, tolerances.
  (Baseline means table lands in todo 12.)
