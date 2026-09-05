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

<!-- TODO-12: baseline means table -->
