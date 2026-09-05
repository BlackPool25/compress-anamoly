# Architecture (skeleton)

Modular harness layout at repo root:

- `harness/datasets/` — 4-morphology seeded generator + UCR loader with checksum/cache
- `harness/codecs/` — R0 raw, R3 quantization, R4 symbolic
| R4 SAX PAA=16 | PARKED — no code path; PAA window frozen at 8 (A=8) per R4 spec |
- `harness/detectors/` — PCA, Isolation Forest, D4 direct-symbolic
- `harness/metrics/` — frozen threshold evaluator, trailing alignment, VUS-PR
- `harness/results/` — run outputs

Details land in later tasks. No logic yet — skeleton only.

## PLAN-FREEZE register

### Rhythm phase-jitter title-vs-formula decision (Task 3, frozen)
TASK_01 titles morphology 2 "Rhythmic Arrhythmia (Period Dilation & Phase
Jitter)" but its exact formula specifies only frequency halving
(`f0 -> 0.5*f0` on `t[1300:1450]`, `f0 = 0.02` frozen, `T0 = 50` samples)
with no phase-jitter term. Decision: implement the formula exactly — no
jitter — and assert the noiseless base within `[-1.0, 1.0]` before adding
`N(0, 0.05)` noise (noisy bound `1.3`). Rationale: the frozen assertions and
the period-doubling acceptance test are only well-defined on the exact
formula; an invented jitter term would unfreeze the plan. Reversible: jitter
can be added later as an explicit, separately-tested term without touching
the frozen base.
