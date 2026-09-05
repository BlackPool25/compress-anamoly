# Architecture

Phase-1 harness: 5 datasets x 4 codec rungs x 3 detectors x 2 paths.
Per-dataset means decide; nothing is ever pooled across morphologies.

## Cell routing matrix

Every dataset in [Spike, Rhythm, Drift, Chaos, UCR] runs:

| Codec | Payload | Detector | Path | Note |
|---|---|---|---|---|
| R0 | raw float32 bytes | PCA, IF | A | lossless baseline, ratio 1.0 |
| Q8 | 8-B header + uint8 codes | PCA, IF | A | R3a uniform 8-bit |
| Q4 | 8-B header + nibble-packed codes | PCA, IF | A | R3b uniform 4-bit, high-nibble-first |
| R4 | 1 byte per SAX token | D4 | B | direct-on-symbols, no decompression |
| R4-decode | Path-A reconstruction | PCA only | A | Drift only; R4-decode to IF is NOT run |

Path A = decode to floats, score with a classical detector under the frozen
tau. Path B = D4 scores R4 int-token streams directly (`decode_tokens`
semantics frozen). Chaos and UCR rows are reported but excluded from every
bump/cliff assert. Aggregation is the arithmetic mean over seeds 42-46 with
sample std alongside; tables print per dataset, never pooled.

## PLAN-FREEZE register

Frozen decisions. Change requires a plan amendment, never a hot-fix.
Columns: decision | value | rationale | reversible-by.

| Decision | Value | Rationale | Reversible-by |
|---|---|---|---|
| R4 SAX params | PAA=8, A=8 | Spec band 16-32x holds at nominal 8000B/250B = 32x (float32); widened ±2 to [14.0, 34.0] for morphology/dtype variation | Plan amendment + new ratio band + tests |
| Seed budget | 42, 43, 44, 45, 46 | 5-seed mean+std per cell, disclose-N; threaded `default_rng(seed)`, no global RNG | Widening the seed set (means only get tighter) |
| Tau key | p99 of R0-train scores per (seed, dataset, detector), frozen across codecs | Train-only calibration; reusing the same tau object across codecs is what makes codec comparison honest | Nothing: retuning tau per codec would unfreeze the experiment |
| D4 tau | p99 of R4-train token scores | D4 rejects float input by dtype gate, so R0 scores are unscoreable; still train-only per (seed, dataset, detector) | Nothing (documented deviation, vacuous freeze: D4 has one codec) |
| Alignment | trailing: window i covers [i, i+w), scored at i+w-1; first w-1 points = train-median fill cached at fit | Spec pins one mapping; leading-mapping tests must NOT match; cached fill keeps `score` side-effect free | Nothing: remapping would rewrite every expected value |
| Rhythm jitter | no jitter term (formula implemented exactly) | Spec title names phase jitter but its formula defines none; invented terms unfreeze the frozen asserts (`max\|y\|<=1.0` noiseless, `<=1.3` noisy) and the period-doubling test | Adding an explicit, separately-tested jitter term |
| Spike sigma order | sigma = std of noisy train slice pre-injection; uplift exactly 3.5*sigma | Order frozen so tests recompute nominal independently (rtol=1e-9) | Nothing (order is the definition) |
| Canonical dtype | float32 cast immediately after load, before any encode | Else R0 ratio reads 2.0 and SAX ratio 64x raises outside its band | Nothing |
| R4 truncation | R4 path only: truncate series+labels to floor(N/8)*8 | `encode` raises unless len%8==0; UCR 79795->79792 (train 31912, test 47872); synthetics already multiples of 8 | Nothing |
| Q4 trim | every decode trimmed/padded to raw N, asserted | Q4 carries no length field so odd-N decode returns N+1 | Nothing |
| D4 order | 1st-order 8x8 `P(t\|t-1)`, +1e-5 smoothing, row-normalized | Spec label says "2nd-order" but its own formula conditions on one prior token; the formula as written IS 1st-order (label error, noted in-module) | Plan amendment + 8x8x8 tensor + new tests |
| D4 position 0 | uniform prior `-log(1/8+eps)` | Stationary prior would couple train dynamics into position 0; uniform keeps it deterministic | Nothing (pinned by test) |
| VUS window | VUS_WINDOW=64, decoupled from detector w=32 | Anomaly lengths span 20-500 pts; 64 sits at mid-scale | Nothing (sensitivity probe documents robustness, never gates) |
| Q8 ratio band | [3.75, 4.15] | Header math 8000/2008 ≈ 3.98; spec ≈3.9-4.0x ±0.2 | Nothing |
| Q4 ratio band | [7.7, 8.1] | Header math 8000/1008 ≈ 7.94; spec ≈7.8-8.0x | Nothing |
| RMSE-only | per-cell RMSE logged; no PRD column | PRD is RMSE normalized by signal range and adds no decision signal | Adding a column (report-only) |
| GPU model | `"cpu"` string per row | CPU-only phase; shared-harness contract field | Real GPU runs |
| Network | timeout 30s, 3 retries; UCR pinned by sha256+bytes, cache-first | Frozen fetch behavior; mismatch deletes cache and raises; offline without cache raises loudly, never substitutes synthetic | Nothing |
| UCR split | train = first 40% verified anomaly-free; exactly one contiguous anomaly fully in test 60% | Split check runs BEFORE the cut; violation triggers the reselection rule (old row SUPERSEDED, new ACTIVE row) | Reselection rule (file keeps exactly one ACTIVE row) |
| Chaos noise order | full-length N(0,0.05) then segment overwritten by second `rng.normal` call | Order-dependent seeding; tests pin behavior, not stream | Nothing |
| Drift ramp | `0.003*(t-1200)` on [1200:1700], max delta +1.5 | Exact formula; endpoint tolerance ±0.15 is noise-driven | Nothing |
| Rhythm base | f0=0.02 frozen (T0=50); anomaly halves frequency on [1300:1450] | Exact formula; zero-crossing period test on equal-length clean window [800:950] | Nothing |

## PARKED register

No code path exists for any parked item (scope-grep clean). Re-entry needs
fresh evidence plus a plan amendment, never a drive-by implementation.

| Item | Status | Re-entry condition |
|---|---|---|
| R4 SAX PAA=16 | Parked; window frozen at 8 | Evidence that PAA=8 hides decision-relevant short anomalies AND PAA=16 preserves them inside a newly derived ratio band, with band + tests amended first |
| PA%K secondary | Parked; never primary, never replacing point/event-F1 | Report alongside (not instead of) the frozen four, with the K budget and N disclosed per cell |
| PATE | Rejected for Phase 1 | New gaming-proof showing threshold-estimation cannot inflate scores; until then the frozen-tau rule stands |

## TRACEABILITY appendix

Round-1 metis session: `ses_f8fae7843ffeoy2GBGdJQUpSzN`. The session text
is not re-quoted here; each row states the tag's plan-cited topic honestly
marked per plan citation, plus where it is frozen in this repo.

| Tag | Plan-cited topic (per plan citation) | Frozen in |
|---|---|---|
| METIS-F01 | TASK_01 section 3 tree relocated to this repo root (never under the research dir) | repo layout, README |
| METIS-F02 | `.omo/` + `.serena/` gitignore rules with plan/draft negations | `.gitignore` |
| METIS-F03 | UCR freeze row pinned before loader logic; no silent fallback | `dataset-freeze.csv`, `harness/datasets/ucr_loader.py` |
| METIS-F04 | R4-only routing (D4 on tokens only); Chaos+UCR excluded from asserts | `harness/detectors/direct_symbolic.py`, runner, eval gates |
| METIS-F05 | No lossy tolerance claims beyond measured ratios; no tweaks to force asserts | codec ratio bands, `eval/check_expected.py` inequalities |
| METIS-F06 | Tau never recomputed on test or per codec | `harness/metrics/frozen_evaluator.py`, runner |
| METIS-F07 | Freeze PAA=8/A=8; no SPARTAN/QABBA/Machete variants | `harness/codecs/symbolic.py`, PARKED register |
| METIS-F08 | Seeded RNG everywhere; multi-seed means | generator, SEEDS=[42..46], `aggregate` |
| METIS-F09 | No PATE in eval | `eval/check_expected.py`, PARKED register |
| METIS-F10 | Exact determinism (byte-identical reruns), no "negligible difference" weasel | detector + codec tests |
| METIS-F11 | No thresholding inside detectors; tau-key reuse; stratified asserts | detectors (scores only), runner, eval gates |
| METIS-F12 | Payload-bytes contract (LE byte format, ratio bands) | `harness/codecs/*.py` |
| METIS-F13 | Seeded generator, no unseeded randomness | `harness/datasets/generator.py` |
| METIS-F14 | Python >=3.11 pin | `pyproject.toml` |
| METIS-F15 | Offline without cache raises; never substitutes synthetic | `ucr_loader.load_series`, runner `--offline` |
| METIS-F16 | Docs todo changes no behavior (docstrings + docs + sweep only) | this commit scope |
| METIS-F17 | Chaos+UCR reported but excluded from bump/cliff asserts; exact matrix | runner matrix, eval gates |
| METIS-F18 | PA-F1 / affiliation / VUS-ROC ban | PA-ban grep, `docs/METRICS.md` gameability note |

MASTER-REPORT CONSTRAINTS (Master PDF section numbers per plan citation;
no section numbers are invented here):

| Constraint | Frozen in |
|---|---|
| Frozen thresholds | tau p99 rule, `frozen_evaluator` + runner |
| PA-F1 gaming proof | PA-ban + METRICS.md gameability note |
| Stratify-never-pool | per-dataset tables; `aggregate` groups by (dataset, codec, detector, path) |
| Dataset-freeze | `dataset-freeze.csv` (series, url, sha256, bytes, license, split_rule) |
| Bump-never-clipped | gate (a): Q8>R0 reported as bonus, never required |
| Boring-flat honest | gate (e): HONEST NO-KNEE exits 0 instead of forcing a knee |
| Hidden-knee audit | `build_freeze_audit`, report-only, never gates |
| RMSE-only freeze | `rmse` column; no PRD (normalized RMSE adds no decision signal) |
