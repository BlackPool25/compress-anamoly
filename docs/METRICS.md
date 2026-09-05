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

## Metric Dynamics: Sliding Windows & Symbolic Quantization

### Window Boundary Dilation Effect on Point-F1
When a detector operates with sliding window $w=32$ and trailing point alignment,
any window containing an anomaly point marks its trailing index as anomalous.
For an anomaly segment of true length $L$, the window overlaps the anomaly across
$L + w - 1$ consecutive positions.
If the detector triggers across all overlapping windows with zero spurious alarms
elsewhere in the sequence, the maximum achievable pointwise precision is:
$$\text{Precision}_{\text{max}} = \frac{L}{L + w - 1}$$
For the synthetic Spike morphology ($L = 20, w = 32$):
$$\text{Precision}_{\text{max}} = \frac{20}{20 + 32 - 1} = \frac{20}{51} \approx 0.392$$
Consequently, unadjusted Point-F1 is mathematically capped around $\approx 0.56$
even for a perfect detector. This explains why Spike Point-F1 settles near $0.45-0.54$
while Event-F1 reaches $0.80-1.00$ and VUS-PR reaches $0.71-0.78$.

### Symbolic Temporal Resolution (PAA=8 Chunking)
In the R4 path, SAX applies Piecewise Aggregate Approximation ($PAA=8$), compressing
8 time steps into a single symbolic token. When Direct-Symbolic (D4) flags an
anomaly token and upsamples it by $\times 8$ back to the original time resolution,
boundary alignment is quantized to multiples of 8.
This slight boundary mismatch slightly lowers Point-F1 ($0.7046$ vs $0.8854$),
yet Event-F1 captures the true sequence detection capability ($0.9121$ vs $0.7133$,
a +20pp improvement over uncompressed PCA). Event-F1 is therefore the operational
ground truth for symbolic sequence evaluation.

## TCN Autoencoder: Trailing Squared-Error Scoring

The Phase-2 Temporal Convolutional Network (TCN) Autoencoder detector scores input sequences via point-level reconstruction error:

1. **Window Alignment:** Operates with window length $w=64$ and stride $1$. For each window $W_i = [x_i, \dots, x_{i+w-1}]$, the network outputs reconstruction $\hat{W}_i$.
2. **Trailing Alignment:** Point assignment maps the squared error of the trailing window position:
   $$\text{score}(i + w - 1) = (x_{i+w-1} - \hat{x}_{i+w-1})^2$$
   To ensure full length coverage without boundary leakage, the leading $w - 1 = 63$ positions are imputed with the median score computed on the R0 nominal training series.
3. **Threshold Calibration:** In strict adherence to the frozen-tau protocol, $\tau_{\text{TCN}}$ is calibrated once per (seed, dataset) as the 99th percentile of nominal R0 training scores:
   $$\tau = \text{percentile}(\text{score}(x_{\text{train}}^{\text{R0}}), 99.0)$$
   This identical threshold object is reused across all decoded codecs ($R1, R2a, R2b, Q8, Q4, Q2$), preventing artificial metric inflation.

