# Eval

Gates read MEAN cells (5-seed arithmetic means per dataset, codec, detector,
path) from a 14-col harness CSV and assert inequalities only, never
exact illustration values.

## Expected phenomena

| # | Phenomenon | Cell | Expectation |
|---|---|---|---|
| (a) | Non-collapse | Spike / IF, Q8 vs R0 | Q8 point-F1 >= R0 point-F1 (bump is a reported bonus, never required) |
| (b) | Cliff past R3 | Drift / IF, Q4 vs R0 | Q4 point-F1 <= R0 point-F1 - 0.20 |
| (c) | Symbolic-path cliff | Drift / R4-decode / PCA | event-F1 <= 0.20 |
| (d) | Grammar win | Rhythm R4/D4 vs R0/PCA | R4/D4 point-F1 >= R0/PCA point-F1 - 0.05 |
| (e) | Honest no-knee | any stratum past R3 | no cliff observed -> print HONEST NO-KNEE, exit 0, never force a knee |

## Gate inequalities and tolerances

- (a) `Q8_point_f1 >= R0_point_f1` on (Spike, IF, A). Zero tolerance:
  any regression fails. `Q8 > R0` prints a BONUS line, never gates.
- (b) `Q4_point_f1 <= R0_point_f1 - 0.20` on (Drift, IF, A). Tolerance 0.20
  from the illustration drop (0.75 -> 0.38).
- (c) `event_f1 <= 0.20` on (Drift, R4-decode, PCA, A). Tightened from the
  illustration 0.00 with headroom; a broken symbolic path scoring above
  0.20 fails.
- (d) `R4/D4_point_f1 >= R0/PCA_point_f1 - 0.05` on Rhythm. Tolerance 0.05
  for the cross-detector comparison (symbolic sequence win documented).
- (e) Ordering: (a) and (d) are enforced always; then if NEITHER (b) NOR (c)
  cliff is observed, the run exits 0 with the HONEST NO-KNEE note (each
  `fail_*` fixture keeps one cliff alive, so the rescue cannot mask a real
  cliff). Missing cells fail by name, never bare KeyError.

## Freeze audit (report-only proxy)

`build_freeze_audit` takes the Spike/IF/R0 MEAN tau, shifts it ±5%, and
reports a 0.0pp sensitivity proxy: the CSV carries no point scores, so F1
cannot be recomputed at shifted tau and this is NOT a true retune. It never
gates. It writes `eval/freeze_audit.md` only for the real
`harness/results/baseline_run.csv`, never for fixtures.

## Fixtures and hygiene

`eval/fixtures/pass.csv` passes all gates; `fail_a_collapse`, `fail_b_nocliff`,
`fail_c_symbolic`, `fail_d_rhythm` each break exactly their named gate.
Tests cover all five. PA-ban grep over `harness/` fails the run on
point-adjust, affiliation, or VUS-ROC tokens.

## Baseline means table

## Baseline means table

Full-matrix baseline run: 5 seeds x 36 cells (Spike/Chaos/Rhythm/UCR x 7,
Drift x 8 with the extra R4-decode row) = 180 rows, CPU-only, wall ~40s.
Source: `harness/results/baseline_run.csv` (committed). Tables below are the
runner's verbatim aggregated means (mean ± std over n=5 seeds).

## Chaos
| codec   | detector   | path   |   n | point_f1        | event_f1        | pr_auc          | vus_pr          |
|---------|------------|--------|-----|-----------------|-----------------|-----------------|-----------------|
| Q4      | IF         | A      |   5 | 0.6118 ± 0.0696 | 0.0990 ± 0.0403 | 0.8738 ± 0.0356 | 0.9494 ± 0.0153 |
| Q4      | PCA        | A      |   5 | 0.2442 ± 0.0162 | 0.2863 ± 0.4001 | 0.9601 ± 0.0299 | 0.9889 ± 0.0081 |
| Q8      | IF         | A      |   5 | 0.8478 ± 0.0285 | 0.4029 ± 0.1996 | 0.8910 ± 0.0249 | 0.9614 ± 0.0128 |
| Q8      | PCA        | A      |   5 | 0.8973 ± 0.0128 | 0.7133 ± 0.2785 | 0.9660 ± 0.0196 | 0.9901 ± 0.0058 |
| R0      | IF         | A      |   5 | 0.8486 ± 0.0242 | 0.3293 ± 0.1466 | 0.8909 ± 0.0243 | 0.9612 ± 0.0115 |
| R0      | PCA        | A      |   5 | 0.8989 ± 0.0134 | 0.8000 ± 0.1826 | 0.9660 ± 0.0198 | 0.9901 ± 0.0058 |
| R4      | D4         | B      |   5 | 0.1772 ± 0.0688 | 0.4243 ± 0.1864 | 0.2085 ± 0.0691 | 0.2242 ± 0.0702 |

## Drift
| codec     | detector   | path   |   n | point_f1        | event_f1        | pr_auc          | vus_pr          |
|-----------|------------|--------|-----|-----------------|-----------------|-----------------|-----------------|
| Q4        | IF         | A      |   5 | 0.6957 ± 0.0351 | 0.4181 ± 0.0742 | 0.7621 ± 0.0420 | 0.8191 ± 0.0351 |
| Q4        | PCA        | A      |   5 | 0.5953 ± 0.0060 | 0.7931 ± 0.1969 | 0.6702 ± 0.0041 | 0.7455 ± 0.0059 |
| Q8        | IF         | A      |   5 | 0.7680 ± 0.0329 | 0.8802 ± 0.0899 | 0.8399 ± 0.0174 | 0.8947 ± 0.0118 |
| Q8        | PCA        | A      |   5 | 0.7918 ± 0.0318 | 0.9905 ± 0.0213 | 0.7470 ± 0.0213 | 0.8222 ± 0.0218 |
| R0        | IF         | A      |   5 | 0.7676 ± 0.0339 | 0.8814 ± 0.0724 | 0.8398 ± 0.0173 | 0.8946 ± 0.0118 |
| R0        | PCA        | A      |   5 | 0.7907 ± 0.0321 | 0.9867 ± 0.0298 | 0.7477 ± 0.0212 | 0.8230 ± 0.0218 |
| R4        | D4         | B      |   5 | 0.1983 ± 0.0306 | 0.7749 ± 0.0468 | 0.4682 ± 0.0092 | 0.5125 ± 0.0124 |
| R4-decode | PCA        | A      |   5 | 0.5985 ± 0.0009 | 1.0000 ± 0.0000 | 0.3478 ± 0.0068 | 0.3913 ± 0.0081 |

## Rhythm
| codec   | detector   | path   |   n | point_f1        | event_f1        | pr_auc          | vus_pr          |
|---------|------------|--------|-----|-----------------|-----------------|-----------------|-----------------|
| Q4      | IF         | A      |   5 | 0.7901 ± 0.0051 | 0.2091 ± 0.0792 | 0.7378 ± 0.0470 | 0.8717 ± 0.0236 |
| Q4      | PCA        | A      |   5 | 0.3892 ± 0.0544 | 0.0625 ± 0.0193 | 0.7061 ± 0.0047 | 0.8507 ± 0.0036 |
| Q8      | IF         | A      |   5 | 0.8429 ± 0.0210 | 0.3586 ± 0.1120 | 0.7355 ± 0.0447 | 0.8746 ± 0.0242 |
| Q8      | PCA        | A      |   5 | 0.8854 ± 0.0144 | 0.7133 ± 0.2785 | 0.7040 ± 0.0079 | 0.8487 ± 0.0046 |
| R0      | IF         | A      |   5 | 0.8464 ± 0.0189 | 0.3873 ± 0.1164 | 0.7383 ± 0.0447 | 0.8750 ± 0.0238 |
| R0      | PCA        | A      |   5 | 0.8854 ± 0.0144 | 0.7133 ± 0.2785 | 0.7041 ± 0.0083 | 0.8486 ± 0.0049 |
| R4      | D4         | B      |   5 | 0.7046 ± 0.0186 | 0.9121 ± 0.0592 | 0.6534 ± 0.0189 | 0.7122 ± 0.0254 |

## Spike
| codec   | detector   | path   |   n | point_f1        | event_f1        | pr_auc          | vus_pr          |
|---------|------------|--------|-----|-----------------|-----------------|-----------------|-----------------|
| Q4      | IF         | A      |   5 | 0.1527 ± 0.0526 | 0.0330 ± 0.0229 | 0.3063 ± 0.1050 | 0.7388 ± 0.0827 |
| Q4      | PCA        | A      |   5 | 0.0374 ± 0.0037 | 0.1425 ± 0.0789 | 0.2988 ± 0.0275 | 0.6956 ± 0.0393 |
| Q8      | IF         | A      |   5 | 0.4495 ± 0.0628 | 0.2248 ± 0.1520 | 0.3705 ± 0.0949 | 0.7838 ± 0.0794 |
| Q8      | PCA        | A      |   5 | 0.5411 ± 0.0456 | 0.8333 ± 0.2357 | 0.3106 ± 0.0120 | 0.7135 ± 0.0181 |
| R0      | IF         | A      |   5 | 0.4597 ± 0.0559 | 0.2354 ± 0.1591 | 0.3735 ± 0.0978 | 0.7846 ± 0.0769 |
| R0      | PCA        | A      |   5 | 0.5406 ± 0.0424 | 0.7667 ± 0.2236 | 0.3100 ± 0.0120 | 0.7118 ± 0.0173 |
| R4      | D4         | B      |   5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0166 ± 0.0006 | 0.0456 ± 0.0069 |

## UCR
| codec   | detector   | path   |   n | point_f1        | event_f1        | pr_auc          | vus_pr          |
|---------|------------|--------|-----|-----------------|-----------------|-----------------|-----------------|
| Q4      | IF         | A      |   5 | 0.2688 ± 0.0102 | 0.0782 ± 0.0105 | 0.2256 ± 0.0261 | 0.1848 ± 0.0224 |
| Q4      | PCA        | A      |   5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0168 ± 0.0000 | 0.0171 ± 0.0000 |
| Q8      | IF         | A      |   5 | 0.2657 ± 0.0147 | 0.0701 ± 0.0060 | 0.2078 ± 0.0306 | 0.1731 ± 0.0191 |
| Q8      | PCA        | A      |   5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0172 ± 0.0000 | 0.0178 ± 0.0000 |
| R0      | IF         | A      |   5 | 0.2658 ± 0.0147 | 0.0699 ± 0.0045 | 0.2080 ± 0.0305 | 0.1742 ± 0.0194 |
| R0      | PCA        | A      |   5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0172 ± 0.0000 | 0.0177 ± 0.0000 |
| R4      | D4         | B      |   5 | 0.0106 ± 0.0000 | 0.0206 ± 0.0000 | 0.0144 ± 0.0000 | 0.0147 ± 0.0000 |

## Eval outcome on this CSV

`uv run python eval/check_expected.py --csv harness/results/baseline_run.csv`
exits 1 (honest result, no thresholds touched):

- HONEST NO-KNEE: no cliff past R3 on any stratum (Drift/IF Q4 0.6957
  vs R0 0.7676; Drift/R4-decode/PCA event_f1 1.0000); refusing to force a knee
- FAIL gate (a) NON-COLLAPSE: Spike/IF Q8 point_f1 0.4495 < R0 0.4597
- FAIL gate (d) GRAMMAR: Rhythm R4/D4 point_f1 0.7046 < R0/PCA 0.8854 - 0.05
- Gates (b)/(c) cliff asserts not reached alongside the NO-KNEE note;
  flagged for plan amendment, CSV committed as-is.

Determinism: `--seeds 42 --output /tmp/seed42.csv` (36 rows) matches the
seed-42 subset of the full CSV exactly — PASS. Full-run repeat is
byte-identical (`cmp` clean). `uv run pytest -q`: 84 passed, 1 skipped.
