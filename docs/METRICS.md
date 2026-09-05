# Metrics

Four metrics, each earning its place. One frozen threshold rules them all:
tau = 99th percentile of R0 train scores per (seed, dataset, detector),
reused across codecs, never recomputed.

## Why these four

- **point-F1** — the honest baseline: `y_score >= tau` against binary labels.
  No adjustment, no forgiveness. `0.0` when nothing fires.
- **event-F1** — the operational view: a labeled region is a hit if any point
  inside it fires; a detected segment with zero label overlap is a false
  alarm. Partial overlap counts as a true detection. Answers "did we catch
  the event without crying wolf."
- **PR-AUC** (average precision) — threshold-free ranking quality. Needed
  because tau is frozen, not tuned: a codec can rank well even where the
  frozen cut lands awkwardly.
- **VUS-PR** — buffer-tolerant PR surface from the author reference
  (`vus==0.0.6`, `generate_curve`, VUS-PR is the last tuple element, window
  passed positionally). Rewards near-misses at anomaly boundaries where
  point-F1 sees only misses. VUS-ROC is banned with the rest (see below).

## Gameability note

Point-adjusted F1 inflates to ~0.89 on random scores: firing anywhere inside
a long anomaly region marks the whole region correct, so random firing looks
near-perfect. Affiliation metrics are gameable the same way (loose
affiliation windows forgive far-away detections). ROC inflates under class
imbalance (anomalies are ~1% of points here): the huge nominal pool keeps
false-positive rate tiny no matter how many false alarms fire. All three are
literature-known effects, stated here as the reason for the ban, with no new
experiments run to re-prove them. Enforcement: PA-ban grep over `harness/`
fails on `point_adjust`, `affiliation`, or `vus_roc`; the eval script names
these tokens only to ban them.

## VUS window: why 64, decoupled from w=32

Detector windows (w=32) set the scale of *scoring*; the VUS buffer sets the
scale of *forgiveness*. Coupling them would let a detector choice silently
redefine what counts as "close". Anomaly lengths here span 20-500 points, so
64 sits at mid-scale: small enough that a far-away detection still fails,
large enough that boundary wobble on a 500-point Drift region does not tank
the score. PLAN-FREEZE: do not retune.

Window-sensitivity result (task-2 evidence, spike fixture): VUS-PR at 64 vs
32 differs by 0.0092, far under the 0.10 robustness note. The check is
informational, never a gate. Supporting probes: perfect scores give exactly
1.0; all-zero scores give 0.3347, strictly lower; mismatched label/score
lengths raise.
