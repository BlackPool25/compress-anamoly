# compress-anomaly-phase1-harness - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** A clean, reproducible test lab that compresses synthetic and real sensor-like signals at several strengths, runs three anomaly detectors on them under strict no-peeking rules, and reports exactly where detection breaks — plus the paperwork proving the numbers are honest.

**Why this approach:** Build the measuring instruments first (metrics before everything), freeze every judgment call up front (seeds, thresholds, tolerances), and gate the headline claims on inequality checks rather than eyeballing tables — because the research literature shows this exact kind of benchmark is easy to game by accident.

**What it will NOT do:** It will not build the adaptive router, edge deployment, or neural compressors — those are later phases. It will not chase extra compression schemes beyond the three core rungs, and it will not push anything to the cloud.

**Effort:** Medium
**Risk:** Medium - the trickiest part is pinning down the VUS-PR metric implementation and the real-data download, both handled with fallback rules
**Decisions to sanity-check:** Adding the VUS-PR metric on top of the original spec; freezing the symbolic codec to one fixed setting; using five fixed random seeds; asserting the expected phenomena with tolerances instead of just printing tables.

Your next move: high-accuracy review is running (you asked for the brutal audit) — results below when both reviewers report. Full execution detail follows below.

---

> TL;DR (machine): Medium effort/risk; Phase-1 compression-vs-detection harness with frozen metrics, 5-seed eval gates, and full docs in a fresh local repo.

## Scope
### Must have
- harness/ package per TASK_01 section 3 tree (relocated to /home/shreyas/projects/compress-anamoly/), built in dependency order metrics -> datasets -> codecs -> detectors -> runner (TASK_01:252-257)
- 4 seeded morphologies (exact formulas) + 1 pinned live UCR-250 sample with dataset-freeze.csv row
- R0/R3(Q8/Q4)/R4(SAX PAA=8/A=8) codecs with measured byte ratios; dual Path A decode + Path B tokens
- PCA(w=32)/IF/D4-Markov detectors, all seeded; trailing window-to-point alignment (frozen)
- Frozen-tau evaluator: point-F1 + event-F1 + PR-AUC + VUS-PR (USER-LOCKED DELTA); PA-F1/affiliation/VUS-ROC banned with grep gate
- run_harness.py full 5-seed matrix + Markdown table + harness/results/baseline_run.csv; eval gates (bump/cliff inequalities + honesty rule)
- pytest suite (tests-after), full docs set, local git with per-todo commits, uv.lock committed
### Must NOT have (guardrails, anti-slop, scope boundaries)
- No D2 router, D3 edge/Telegraf, D5 neural, FM pilots, PATE headline, cloud-shadow eval
- No extra codec rungs (Chimp/Elf/ALP/Machete/SPARTAN/rollup) beyond R0/R3/R4; no PAA=16 variant
- No new dependencies beyond pinned set without plan amendment; no GitHub remote/push
- No exact-value asserts on TASK_01 illustration numbers; no pooling metrics across morphologies; no silent synthetic substitution for UCR

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after + pytest framework (user-locked); every todo ships implementation + tests together and runs `uv run pytest -q`
- Evidence: .omo/evidence/task-<N>-compress-anomaly-phase1-harness.log (outside ulw-loop, per plan template fallback); every todo below uses exactly this naming
- Gates: per-todo acceptance commands + todo 10 eval asserts (bump/cliff) + PA-ban grep + final `uv run pytest -q` green + determinism re-run check
## Execution strategy
### Parallel execution waves
- Wave 0: todo 1 (scaffold)
- Wave 1: todo 2 (metrics)
- Wave 2: todos 3, 4 in parallel (generator, ucr_loader)
- Wave 3: todos 5, 6 in parallel (R0/R3, R4)
- Wave 4: todos 7, 8 in parallel (classical detectors, D4)
- Wave 5: todo 9 (runner)
- Wave 6: todo 10 (gates) then todo 11 (docs) then todo 12 (full run)
### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 scaffold | none | 2-8 | none |
| 2 metrics | 1 | 9,10 | none |
| 3 generator | 1 | 9,10 | 4 |
| 4 ucr_loader | 1 | 9,10 | 3 |
| 5 codecs R0/R3 | 1 | 9,10 | 6 |
| 6 codec R4 | 1 | 8,9,10 | 5 |
| 7 PCA/IF | 1,2 | 9,10 | 8 |
| 8 D4 | 6 | 9,10 | 7 |
| 9 runner | 2,3,4,5,6,7,8 | 10,11,12 | none |
| 10 eval gates | 9 | 11,12 | none |
| 11 docs | 9,10 | 12 | none |
| 12 full run | 9,10,11 | none | none |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. Scaffold uv project, layout, git init, docs skeleton
  What to do: In /home/shreyas/projects/compress-anamoly/ create uv project (pyproject.toml: requires-python>=3.11 (runner env is 3.11; lower bound only, no exact pin), numpy==1.26.* (only upper bound in plan: numpy<2.0 per TASK_01), scipy>=1.13, scikit-learn>=1.4, tabulate>=0.9, vus==0.0.6 author reference impl — other upper bounds dropped: resolver breakage risk exceeds benefit, PLAN-FREEZE), directories harness/datasets harness/codecs harness/detectors harness/metrics harness/results tests/ docs/ eval/ eval/fixtures data/cache/, all __init__.py, tests/test_import.py placeholder (`def test_import(): import harness`), .gitignore with EXACT lines: `.omo/evidence/`, `.serena/`, `data/cache/`, `__pycache__/`, `*.pyc`, `.venv/`, plus negation `!.omo/plans/`, `!.omo/drafts/` (plans/drafts ARE committed), README.md quickstart skeleton (README notes: repo root spelling is user-owned — always run from root, package imports as `harness.*` after `uv sync`), docs/ARCHITECTURE.md + docs/METRICS.md + docs/EVAL.md skeletons, dataset-freeze.csv with header series,url,sha256,bytes,license,split_rule. git init -b main + initial commit. VUS dep ships here (fixing Wave0/1 sequencing: todo 2 only WIRES it). uv.lock committed. Must NOT do: no harness code yet; no GitHub remote (local-only); do not place harness/ under the research dir (METIS-F01: TASK_01 section 3 path is RELOCATED to this repo root).
  Parallelization: Wave 0 | Blocked by: none | Blocks: 2,3,4,5,6,7,8
  References: TASK_01_FIRST_RESEARCH_EXPERIMENT.md:47-76 (tree+pyproject, at /home/shreyas/Downloads/OmniLearn/.omnilearn/research/d943471a-compress-before-anomalies-disappear/TASK_01_FIRST_RESEARCH_EXPERIMENT.md), TASK_01:237-250 (uv deps), METIS-F01/F02/F14/F16 (path relocation, .omo/.serena rule, python pin, layout bound)
  Acceptance criteria: `uv sync` succeeds; `uv run python -c "import sys; print(sys.version)"` shows 3.11; `test -f uv.lock`; `uv run pytest -q` passes via placeholder test; `git -C /home/shreyas/projects/compress-anamoly log --oneline | wc -l` >= 1
  QA scenarios: happy `uv sync && uv run pytest -q` (exit 0), Evidence .omo/evidence/task-1-compress-anomaly-phase1-harness.log; failure `git status --porcelain=v1 -b` must show clean tree after commit else FAIL
  Commit: Y | chore(scaffold): uv py3.11 layout, git init, docs skeleton
- [x] 2. Metrics module: frozen threshold, point/event-F1, PR-AUC, VUS-PR
  What to do: Implement harness/metrics/frozen_evaluator.py (calibrate_frozen_threshold(train_scores, percentile=99.0) -> tau; TAU KEY FROZEN: tau computed per (seed,dataset,detector) on R0-train-scores only, frozen across codecs; evaluate_point_f1; evaluate_event_f1 with rule: window scores mapped via harness/metrics/alignment.py::window_to_point_trailing — TRAILING assignment of each window score to its last index, first w-1 points filled with median of that detector's R0 train scores for that seed/dataset (helper LIVES in metrics, todo 7 imports it; PLAN-FREEZE, spec defines no alignment); event hit if >=1 point >= tau inside labeled region, false alarms = detected segments outside labels; evaluate_pr_auc via sklearn.
  VUS: harness/metrics/vus.py (VUS-PR FROZEN: primary impl PyPI `vus==0.0.6` author reference (TheDatumOrg, Apache-2.0): call generate_curve(label, score, window) with module constant VUS_WINDOW=64 — DECOUPLED from detector w=32; scale rationale in code comment (anomaly lengths span 20-500 pts, 64 is mid-scale PLAN-FREEZE) — and return its VUS_PR; FALLBACK if pip install fails: vendor generate_curve from TheDatumOrg/VUS at pinned commit with LICENSE + citation comment, identical wrapper; harness signature vus_pr(y_true: np.ndarray, y_score: np.ndarray) -> float).
  Tests: tests/test_metrics.py covers tau==99th pct; tau-key test (R0-tau object reused on Q8 scores, never recomputed); point-F1 hand-computed case; event hit + false-alarm counting; PR-AUC sanity; tests/test_alignment.py: synthetic window scores pin TRAILING mapping (leading-mapping must NOT match); VUS behavioral conformance (perfect scores -> 1.0 ±1e-6; all-zero scores score strictly lower; mismatched lengths raise; pip-vs-vendored agreement ±1e-6 when both importable; window-sensitivity check VUS_WINDOW=64 vs 32 differ by < 0.10 on spike fixture — documents robustness, never a gate); assert no symbol named point_adjust/point_adjustment exists (PA-F1 ban, METIS-F18). Must NOT do: no PA-F1/affiliation/VUS-ROC functions; tau never recomputed on test or per codec (METIS-F06/F11/F18).
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 2->7 edge (todo 7 imports alignment from here), 9, 10
  References: TASK_01:182-192 (evaluator spec); 04-synthesis/REPORT.md:57 (metric lock VUS-PR+point/event-F1, PA banned); FINAL-REPORT-PHASE3.md:56-76 (PA%K secondary only, disclose-N, PATE rejected); METIS-F06/F11/F18; MASTER-REPORT CONSTRAINT: frozen thresholds, PA-F1 gaming proof
  Acceptance criteria: `uv run pytest tests/test_metrics.py -q` all pass; `uv run python -c "from harness.metrics.vus import vus_pr, VUS_WINDOW; print(VUS_WINDOW)"` prints 64; EITHER `uv run python -c "import vus"` succeeds (pip source) OR harness/metrics/_vendored_vus.py exists with LICENSE header (vendored source) — one source mandatory
  QA scenarios: happy pytest pass, Evidence .omo/evidence/task-2-compress-anomaly-phase1-harness.log; failure mutate tau percentile to 95.0 -> event-F1 test must FAIL (proves freeze sensitivity)
  Commit: Y | feat(metrics): frozen evaluator plus VUS-PR with tests
- [x] 3. Datasets generator: 4 morphologies, seeded, frozen formulas
  What to do: Implement harness/datasets/generator.py with TimeSeriesSample(x: np.ndarray[N=2000], y: np.ndarray binary, meta) + functions make_spike/make_rhythm/make_drift/make_chaos exactly per TASK_01: base y=sin(0.05t)+noise with t=np.arange(2000); spike t[1400:1420] += 3.5*sigma where sigma := std of the nominal train slice (first 800 pts) computed AFTER noise injection but BEFORE anomaly injection (order frozen, deterministic given seed); rhythm base sin(2*pi*f0*t) with FROZEN f0=0.02 (T0=50 samples) and t=np.arange(2000), anomaly t[1300:1450] f0->0.5*f0, NO phase-jitter term (spec L94 title names jitter but L95-97 formula defines none; PLAN-FREEZE: implement formula exactly, record title-vs-formula decision in docs/ARCHITECTURE.md, reversible); BUILD ORDER FROZEN: construct anomaly on the NOISELESS base first and assert max|y|<=1.0 (spec L97 honored strictly), THEN add observation noise; noisy-signal test asserts max|y|<=1.3 (3-sigma noise headroom, documented); drift t[1200:1700] += 0.003*(t-1200); chaos t[1350:1500] noise std 0.05->0.40; split train=first 800 (Y=0), test=rest; all RNG via np.random.default_rng(seed) threaded through every function (no global np.random).
  Tests: tests/test_generator.py covers N/split/labels per morphology; seed reproducibility (same seed byte-identical, different seed differs); spike amplitude == 3.5*train_std (recomputed independently in test); rhythm period doubles (zero-crossing check); drift endpoint delta ≈ +1.5 ±0.15 (PLAN-FREEZE default tolerance, noise-driven); chaos variance ratio in [40,90] (spec L105 implies (0.40/0.05)^2=64; PLAN-FREEZE band). Must NOT do: no UCR code here; no unseeded randomness (METIS-F08/F10/F13).
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 9,10 (parallel with 4)
  References: TASK_01:82-106 (morphology formulas); TASK_01:270 (seeded); METIS-F08/F10/F13; MASTER-REPORT CONSTRAINT: stratify-never-pool (labels per morphology kept)
  Acceptance criteria: `uv run pytest tests/test_generator.py -q` all pass
  QA scenarios: happy pytest pass, Evidence .omo/evidence/task-3-compress-anomaly-phase1-harness.log; failure run with seed 7 vs 7 twice -> byte-identical arrays else FAIL; seed 7 vs 8 -> differ else FAIL
  Commit: Y | feat(datasets): seeded 4-morphology generator with tests
- [x] 4. UCR series pin + loader with checksum, cache, freeze record
  What to do: PHASE A (pin first, no code): resolve ONE stable labeled UCR time-series-anomaly series (UCR Time Series Anomaly Archive), record series,url,sha256,bytes,license,split_rule as the FIRST row of dataset-freeze.csv and verify with `sha256sum` of actually downloaded bytes — the pin is part of THIS todo's acceptance, never deferred. PHASE B: implement harness/datasets/ucr_loader.py: cache-first at data/cache/<name>; download with timeout=30s + 3 retries (PLAN-FREEZE network defaults); verify SHA256 against pinned value, on mismatch delete + raise; parse to TimeSeriesSample with train=first 40% VERIFIED anomaly-free (loader asserts zero labeled anomalies in train slice AND exactly one contiguous labeled anomaly region fully inside the test 60%, else raise with message; split check runs BEFORE the train/test cut; split_rule recorded verbatim in freeze row; RESELECTION RULE: if the pinned series violates the anomaly check, pick the next UCR candidate, mark the old freeze row SUPERSEDED (keep it), and pin a new ACTIVE row — the freeze file always ends with exactly one ACTIVE UCR row); offline without cache -> raise clear error (never silently substitute synthetic). Tests test_ucr_loader.py with mocked HTTP: cache-hit path; checksum-mismatch raises; offline-no-cache raises; freeze-row schema validated. Must NOT do: no loader logic before the freeze row exists; no silent fallback (METIS-F03/F15).
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 9,10 (parallel with 3)
  References: TASK_01:16 (1 real UCR sample), TASK_01:60 (ucr_loader), TASK_01:200 (UCR-001 in matrix); METIS-F03/F15; MASTER-REPORT CONSTRAINT: dataset-freeze.csv (sha256, bytes, licences)
  Acceptance criteria: `uv run pytest tests/test_ucr_loader.py -q` all pass; dataset-freeze.csv contains one live-verified row (real URL + matching SHA256 + bytes + license, no placeholders)
  QA scenarios: happy mocked pytest pass, Evidence .omo/evidence/task-4-compress-anomaly-phase1-harness.log; failure corrupt one cached byte -> loader must raise checksum error else FAIL
  Commit: Y | feat(datasets): pinned UCR series plus loader with cache and freeze record
- [x] 5. Codecs R0 + R3: lossless and scalar quantization
  What to do: Implement harness/codecs/base.py (BaseCodec verbatim per TASK_01 incl. get_ratio), lossless.py (R0 raw float32 bytes, ratio==1.0), quantization.py (BYTE FORMAT FROZEN: little-endian throughout; R3a Q8: 8-byte header [min,max float32 LE] + uint8 codes; R3b Q4: same header + nibble-packed bytes HIGH-nibble-first, odd-N low-nibble zero-padded; min==max edge -> all-zero codes, decode returns constant, ratio still reported; decode reconstructs floats). Tests test_codecs_r0_r3.py: R0 roundtrip bit-exact + ratio==1.0; Q8 realized ratio in [3.75,4.15] (header math 8000/2008≈3.98; spec L139 ≈3.9-4.0x; band = spec ±0.2 max with derivation comment); Q4 in [7.7,8.1] (8000/1008≈7.94; spec L140 ≈7.8-8.0x); decode shapes/dtypes; determinism (encode twice identical); header corruption raises; min==max edge returns constant. Must NOT do: no lossy tolerance claims beyond measured ratios (METIS-F05/F12).
  Parallelization: Wave 3 | Blocked by: 1 | Blocks: 9,10 (parallel with 6)
  References: TASK_01:110-140 (base class + R0 + R3 spec); TASK_01:139-140 (ratio targets); METIS-F12 (payload bytes contract)
  Acceptance criteria: `uv run pytest tests/test_codecs_r0_r3.py -q` all pass
  QA scenarios: happy pytest pass, Evidence .omo/evidence/task-5-compress-anomaly-phase1-harness.log; failure flip one payload byte -> decode must differ or raise else FAIL
  Commit: Y | feat(codecs): R0 lossless and R3 Q8/Q4 with tests
- [x] 6. Codec R4: SAX/PAA symbolic with dual Path A/B
  What to do: Implement harness/codecs/symbolic.py with FROZEN params PAA window=8, alphabet A=8 ('a'-'h'): SAX NUMERICS FROZEN — per-series z-normalization with TRAIN mean/std only (train stats frozen at fit/encode of train slice, reused on test; no test leakage), Gaussian breakpoints via scipy.stats.norm.ppf(linspace(1/8,7/8,7)), Path-A reconstruction = segment bin midpoints mapped back with train mean/std (piecewise-constant float array), encode(x)->bytes packs one byte per token, decode(payload)->Path-A floats, decode_tokens(payload)->int token array (Path B); get_ratio via len(payload) (expect ≈32x at PAA=8: 8000B/250B; assert ratio in [14.0,34.0] else raise with message; spec L146 band 16-32x ±2 tolerance, PLAN-FREEZE rationale). Tests test_codecs_r4.py: token alphabet range; decode length == input length; Path A piecewise-constant over windows of 8; Path B length == N/8; ratio in band; determinism; decode_tokens/decode consistency; train/test norm separation (test encoded with train stats object). PAA=16 is PARKED (no code path, docs/ARCHITECTURE.md parked-row). Must NOT do: no PAA=16 variant, no SPARTAN/QABBA/Machete (parked, METIS-F07).
  Parallelization: Wave 3 | Blocked by: 1 | Blocks: 8,9,10 (parallel with 5)
  References: TASK_01:141-146 (R4 spec, ratio 16-32x); TASK_01:40-43 (dual paths); METIS-F07 (freeze PAA=8/A=8) + F12
  Acceptance criteria: `uv run pytest tests/test_codecs_r4.py -q` all pass
  QA scenarios: happy pytest pass, Evidence .omo/evidence/task-6-compress-anomaly-phase1-harness.log; failure alphabet size changed to 4 in a scratch copy -> cliff gate in todo 10 must show degradation (documents cardinality sensitivity)
  Commit: Y | feat(codecs): R4 SAX PAA-8 with dual paths and tests
- [x] 7. Detectors PCA + IsolationForest with frozen window alignment
  What to do: Implement harness/detectors/base.py (BaseDetector fit/score verbatim), pca_detector.py (window w=32 stride=1, PCA(n_components=2, random_state=seed), score = window reconstruction SSE), isolation_forest.py (window w=32, IsolationForest(n_estimators=100, contamination=0.01, random_state=seed), score = -score_samples); import window_to_point_trailing from harness.metrics.alignment (defined in todo 2; PLAN-FREEZE home). All constructors take seed. Tests test_detectors_classical.py: score length == test length; spike series scores peak inside anomaly (argmax in labels); EXACT reproducibility (same seed twice -> byte-identical scores); NaN-free; alignment covered by tests/test_alignment.py shipped in todo 2 (synthetic scores pin TRAILING vs leading). Must NOT do: no thresholding inside detectors (scores only); no global RNG; no "negligible difference" weasel for determinism (METIS-F10/F11).
  Parallelization: Wave 4 | Blocked by: 1 | Blocks: 9,10 (parallel with 8)
  References: TASK_01:150-176 (detector specs); TASK_01:167 (w=32 stride 1); METIS-F10/F11
  Acceptance criteria: `uv run pytest tests/test_detectors_classical.py -q` all pass
  QA scenarios: happy pytest pass, Evidence .omo/evidence/task-7-compress-anomaly-phase1-harness.log; failure all-zero input -> scores finite (no NaN/inf) else FAIL
  Commit: Y | feat(detectors): PCA and IsolationForest with trailing alignment
- [x] 8. Detector D4: direct-on-symbols Markov (R4 only)
  What to do: Implement harness/detectors/direct_symbolic.py: fit(token_train: int array) builds a 1st-ORDER Markov transition matrix P(t|t-1) with +1e-5 smoothing (spec L176 says "2nd-order" but its own formula conditions on ONE prior token; PLAN-FREEZE: implement the formula as written = 1st-order, note the label mismatch in a code comment); score(token_test) = -log(P+1e-5) per token; resample token scores to sample length by repeating each x8 (PAA=8 inverse, frozen); constructor takes NO seed (fully deterministic, no stochastic op). Wire ONLY to R4 token streams (runner enforces; unit test asserts ValueError if fed float series). Tests test_direct_symbolic.py: uniform token stream -> near-constant scores; planted rare-transition -> spike at that token; resampled length alignment; byte-identical reruns; float-input rejection. Must NOT do: never decode inside D4; no use outside R4 (METIS-F04).
  Parallelization: Wave 4 | Blocked by: 6 | Blocks: 9,10 (parallel with 7)
  References: TASK_01:177-178 (D4 spec); TASK_01:143-145 (tokens); METIS-F04 (R4-only routing)
  Acceptance criteria: `uv run pytest tests/test_direct_symbolic.py -q` all pass
  QA scenarios: happy pytest pass, Evidence .omo/evidence/task-8-compress-anomaly-phase1-harness.log; failure feed raw floats -> ValueError else FAIL
  Commit: Y | feat(detectors): D4 Markov-on-tokens with tests
- [x] 9. Runner: full matrix, routing, CSV + Markdown table
  What to do: Implement harness/run_harness.py with CLI `--smoke` (1 seed=42, spike only, R0+Q8, detectors IF+D4-routing per matrix) + `--seeds 42,43` subset override + `--offline` (skip UCR cell, require cache for smoke of cached rows, else loud error) and full mode over SEEDS=[42,43,44,45,46] (PLAN-FREEZE seed budget, disclose-N). EXACT CELL MATRIX (dataset x codec x detector x path): every dataset in [Spike,Rhythm,Drift,Chaos,UCR] x R0->[PCA(A),IF(A)] + Q8->[PCA(A),IF(A)] + Q4->[PCA(A),IF(A)] + R4->[D4(B)] + R4-decode->[PCA(A)] on Drift only (expected-table row 11; R4-decode->IF NOT run — matrix frozen); tau per (seed,dataset,detector) calibrated on R0-train-scores only, frozen across codecs; per-seed CSV rows; aggregation = arithmetic MEAN over SEEDS per cell (spread = sample std, reported alongside, asserts read means); write harness/results/baseline_run.csv with EXACT 14 columns dataset,codec,payload_bytes,realized_ratio,detector,path,seed,point_f1,event_f1,pr_auc,vus_pr,tau,rmse,gpu_model (rmse = decode-vs-raw RMSE where a decode exists, empty string for D4-direct rows; PRD NOT logged: RMSE-only freeze, PRD is RMSE normalized by signal range and adds no decision signal — recorded in ARCHITECTURE register; gpu_model = platform string, "cpu" for this CPU-only phase — shared-harness contract fields per candidate-designs:4); print per-dataset Markdown tables to stdout (never pooled). Smoke test test_runner_smoke.py runs --smoke and asserts header == 14 expected names + minimum row count. Must NOT do: no pooling across morphologies; Chaos+UCR rows reported but excluded from bump/cliff asserts (METIS-F04/F05/F08/F11/F17).
  Parallelization: Wave 5 | Blocked by: 2,3,4,5,6,7,8 | Blocks: 10,11,12
  References: TASK_01:196-205 (runner + CSV); TASK_01:208-224 (expected table incl. row 11 R4-decode PCA); 02-hypotheses/candidate-designs.md:4 (per-cell log incl. RMSE/PRD + GPU model); METIS-F04/F05/F08/F11/F17
  Acceptance criteria: `uv run python harness/run_harness.py --smoke` exits 0 and writes CSV whose header line equals the 14 names above; `uv run pytest tests/test_runner_smoke.py -q` passes
  QA scenarios: happy smoke run, Evidence .omo/evidence/task-9-compress-anomaly-phase1-harness.log; failure run with `--offline` and empty data/cache -> UCR cell must error loudly with cache message (no silent synthetic swap) else FAIL
  Commit: Y | feat(runner): matrix runner with routing and CSV output
- [x] 10. Eval gates and hygiene audits
  What to do: Implement eval/check_expected.py reading the MEAN-aggregated CSV and asserting: (a) NON-COLLAPSE gate on Spike/IF: Q8 point_f1 >= R0 point_f1 (strictly no regression; denoising bump = Q8 > R0 reported as bonus, never clipped, never required — honest boring-rule); (b) CLIFF gate on Drift/IF: Q4 point_f1 <= R0 point_f1 - 0.20 (PLAN-FREEZE tolerance from illustration 0.75->0.38); (c) CLIFF gate on Drift/R4-decode/PCA: event_f1 <= 0.20 (tightened from illustration 0.00 with headroom; a broken symbolic path scoring above 0.20 FAILS); (d) GRAMMAR gate on Rhythm: R4/D4 point_f1 >= R0/PCA point_f1 - 0.05 (expected-table rows 5 vs 7: symbolic sequence win documented, small tolerance for cross-detector comparison); (e) boring-flat honesty: if no gate triggers past R3 on any stratum, print HONEST NO-KNEE note and exit 0 (never force a knee); PA-F1 ban grep (fail if point_adjust/affiliation/vus_roc appear in harness/); frozen-vs-retuned audit report-only: recompute tau on test for Spike/IF, log hidden-knee delta pp to eval/freeze_audit.md (never gate). Ship concrete fixtures eval/fixtures/pass.csv + eval/fixtures/fail_a_collapse.csv + eval/fixtures/fail_b_nocliff.csv + eval/fixtures/fail_c_symbolic.csv + eval/fixtures/fail_d_rhythm.csv with correct 14-col schema. Tests test_eval.py over all five fixtures. Must NOT do: no exact-value matching on illustration numbers (inequalities only); no PATE (METIS-F09/F18).
  Parallelization: Wave 6 | Blocked by: 9 | Blocks: 12
  References: TASK_01:208-226 (phenomena), TASK_01:267-274 (DoD); MASTER-REPORT CONSTRAINTS: bump never clipped, boring-flat honest, hidden-knee audit, PA ban; METIS-F09/F18
  Acceptance criteria: `uv run python eval/check_expected.py --csv eval/fixtures/pass.csv` exits 0; each of the four fail fixtures exits nonzero naming the failed assert; `uv run pytest tests/test_eval.py -q` passes
  QA scenarios: happy fixture pass, Evidence .omo/evidence/task-10-compress-anomaly-phase1-harness.log; failure fixture with spike Q8 collapse -> assert (a) must FAIL loudly else FAIL
  Commit: Y | feat(eval): bump-cliff gates plus hygiene audits
- [x] 11. Docs finalize and docstring sweep
  What to do: Finalize README (quickstart: uv sync, smoke, full, eval), docs/ARCHITECTURE.md (routing matrix, PLAN-FREEZE register: PAA=8/A=8, seeds, tau key, alignment, jitter/sigma/amplitude decisions with rationale; PARKED register: PAA=16, PA%K secondary, PATE — each with re-entry condition; TRACEABILITY appendix mapping every METIS-F## tag in this plan to its round-1 metis finding in ses_f8fae7843ffeoy2GBGdJQUpSzN and every MASTER-REPORT CONSTRAINT to its Master PDF section), docs/METRICS.md (gameability note: PA-F1 random->0.89, affiliation gamed, ROC inflates; VUS choice + VUS_WINDOW=64 decoupled rationale + window-sensitivity result), docs/EVAL.md (expected table + gate inequalities + tolerances), every public function with docstrings; implement eval/check_docstrings.py here (stdlib ast walk over harness/, fails listing undocumented public functions); docstring-coverage check `uv run python eval/check_docstrings.py` (fails on missing). F2 scope gate: `grep -rniE "paa.*16|pa%k|pate|affiliation|point_adjust" harness/ eval/ || true` must return empty (parked items have no code path). Must NOT do: no new behavior; docs only (METIS-F16).
  Parallelization: Wave 6 | Blocked by: 9,10 | Blocks: 12
  References: TASK_01:55 (README); METIS-F16
  Acceptance criteria: README + 3 docs exist; `uv run python eval/check_docstrings.py` exits 0; `uv run pytest -q` still green
  QA scenarios: happy docs + sweep pass, Evidence .omo/evidence/task-11-compress-anomaly-phase1-harness.log; failure strip one docstring -> sweep must FAIL else FAIL
  Commit: Y | docs: finalized set plus docstring sweep
- [x] 12. Full multi-seed matrix run, results commit
  What to do: Run `uv run python harness/run_harness.py` (all 5 seeds x exact todo-9 matrix; CPU-only, allow long runtime), run eval gates, save harness/results/baseline_run.csv + aggregated means table into docs/EVAL.md appendix, final `uv run pytest -q` green, commit results. If UCR fetch fails (offline): write eval/ucr_status.md = FETCH-FAILED with reason, run matrix WITHOUT the UCR dataset (no placeholder rows, schema unchanged), eval exits 0 with note. Must NOT do: ZERO code changes in this todo — a crash stops the run and is reported back for a plan amendment, never hot-fixed here; no threshold/metric tweaks to force asserts (METIS-F05/F18).
  Parallelization: Wave 6 (last) | Blocked by: 9,10,11 | Blocks: none
  References: TASK_01:259-264 (verification run); TASK_01:267-274 (DoD); METIS-F05/F18
  Acceptance criteria: harness/results/baseline_run.csv exists with 5-seed rows; `uv run python eval/check_expected.py` exits per honest rules; `uv run pytest -q` all green
  QA scenarios: happy full run, Evidence .omo/evidence/task-12-compress-anomaly-phase1-harness.log (includes stdout table + timing); failure re-run with `--seeds 42` -> rows identical to seed-42 subset of full run (determinism) else FAIL
  Commit: Y | chore(results): baseline run CSV, eval outcome

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [x] F1. Plan compliance audit: every TASK_01 DoD item (TASK_01:267-274) + USER-LOCKED DELTAS (VUS-PR, pinned UCR, multi-seed, asserts) traced to a todo; fail on any untraced item
- [x] F2. Code quality review: uv.lock committed (`test -f uv.lock` + pinned vus==0.0.6 present), no unpinned deps, no unseeded RNG (grep np.random\. + random\. without seed), docstrings on all public functions (sweep script green), PA-ban grep clean
- [x] F3. Real manual QA: fresh `uv sync`, `--smoke` run, full-matrix spot check (Spike/IF Q8 vs R0 non-collapse + Rhythm grammar gate hold on means), CSV 14-col header validated, eval script exit codes verified on all five fixtures
- [x] F4. Scope fidelity: no D2/D3/D5/FM/PATE/extra-rung/remote code present; Chaos+UCR excluded from asserts; per-dataset (never pooled) reporting confirmed
## Commit strategy
- Local-only git (no remote): one commit per todo (12 commits: chore(scaffold), feat(metrics), feat(datasets)x2, feat(codecs)x2, feat(detectors)x2, feat(runner), feat(eval), docs, chore(results)); conventional-commit messages exactly as listed per todo; clean tree verified after each commit; .gitignore covers .omo/evidence/ .serena/ data/cache/ __pycache__/ .venv (plans/drafts committed via negations)
## Success criteria
- `uv run pytest -q` fully green; `uv run python harness/run_harness.py` completes 5-seed matrix and writes harness/results/baseline_run.csv
- Eval gates: Spike/IF non-collapse (a) + Drift cliff asserts (b,c) + Rhythm grammar gate (d) pass on means, or HONEST NO-KNEE note where applicable
- dataset-freeze.csv contains live-verified UCR row (todo 4); offline fallback = eval/ucr_status.md FETCH-FAILED with UCR dataset absent (never placeholder rows); docs/ complete (ARCHITECTURE incl. PLAN-FREEZE register, METRICS gameability note, EVAL with tolerances); git log shows per-todo commits on main
