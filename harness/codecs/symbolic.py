"""Rung R4: SAX symbolic codec with PAA window=8, alphabet A=8. FROZEN params.

Pipeline: per-series z-normalization (TRAIN mean/std only) -> PAA means over
non-overlapping windows of 8 -> Gaussian-quantile discretization into 8 bins
-> one byte per token.

Dual execution paths (frozen interface for downstream detectors):
- Path A ``decode``: piecewise-constant float reconstruction (bin midpoints
  mapped back with train mean/std).
- Path B ``decode_tokens``: raw int token sequence, consumed zero-decompression
  by the D4 direct-symbolic detector. Method name and int-token semantics are
  FROZEN — do not rename.
"""

import numpy as np
from scipy.stats import norm

PAA_WINDOW = 8
ALPHABET_SIZE = 8
TOKEN_LETTERS = "abcdefgh"

# Gaussian breakpoints: 7 cut points splitting N(0, 1) into 8 equiprobable bins.
BREAKPOINTS = norm.ppf(np.linspace(1 / 8, 7 / 8, 7))

# Bin midpoints for Path-A reconstruction. Interior bins use the exact midpoint
# of their two bounding breakpoints. Outer bins are unbounded tails, so we
# truncate the Gaussian at +/-3.0 (covers 99.7% of mass) and take the midpoint
# of the truncated interval: lower = (-3.0 + b0) / 2, upper = (b6 + 3.0) / 2.
# Tests pin piecewise-constancy, not these specific values.
_TAIL = 3.0
BIN_MIDPOINTS = np.concatenate(
    [
        [(-_TAIL + BREAKPOINTS[0]) / 2.0],
        (BREAKPOINTS[:-1] + BREAKPOINTS[1:]) / 2.0,
        [(BREAKPOINTS[-1] + _TAIL) / 2.0],
    ]
)

# Spec band 16-32x widened by +/-2 (PLAN-FREEZE): tolerance for morphology- and
# dtype-driven payload variation around the nominal 8000B/250B = 32x (float32).
RATIO_MIN = 14.0
RATIO_MAX = 34.0


class SAXCodec:
    """SAX codec with frozen PAA-8 / A-8 params and train-only normalization."""

    def __init__(self) -> None:
        """Create an unfitted codec; call ``fit`` on the train slice first."""
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, x_train: np.ndarray) -> None:
        """Freeze train mean/std from the nominal train slice."""
        x = np.asarray(x_train, dtype=np.float64)
        self.mean = float(np.mean(x))
        self.std = float(np.std(x))
        if self.std == 0.0:
            self.std = 1e-12

    def encode(self, x: np.ndarray) -> bytes:
        """Compress a float array into one byte per PAA token.

        Normalizes with the FROZEN train stats (never test stats).
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("SAXCodec.encode called before fit (no train stats).")
        x = np.asarray(x, dtype=np.float64)
        if x.size % PAA_WINDOW != 0:
            raise ValueError(f"length {x.size} not divisible by PAA_WINDOW={PAA_WINDOW}.")
        z = (x - self.mean) / self.std
        paa = z.reshape(-1, PAA_WINDOW).mean(axis=1)
        tokens = np.searchsorted(BREAKPOINTS, paa).astype(np.uint8)
        return bytes(tokens)

    def decode(self, payload: bytes) -> np.ndarray:
        """Path A: piecewise-constant float reconstruction, length 8 * n_tokens."""
        tokens = self.decode_tokens(payload)
        vals = BIN_MIDPOINTS[tokens] * self.std + self.mean  # type: ignore[operator]
        return np.repeat(vals, PAA_WINDOW).astype(np.float64)

    def decode_tokens(self, payload: bytes) -> np.ndarray:
        """Path B: raw int token array (0-7), length n_tokens = N / 8."""
        if self.mean is None or self.std is None:
            raise RuntimeError("SAXCodec.decode_tokens called before fit.")
        return np.frombuffer(bytes(payload), dtype=np.uint8).astype(np.int64)

    def get_ratio(self, x_raw: np.ndarray, payload: bytes) -> float:
        """Return raw_bytes / payload_bytes; raise outside [14.0, 34.0]."""
        ratio = np.asarray(x_raw).nbytes / max(len(payload), 1)
        if not (RATIO_MIN <= ratio <= RATIO_MAX):
            raise ValueError(
                f"SAX ratio {ratio:.2f}x outside frozen band "
                f"[{RATIO_MIN}, {RATIO_MAX}] (raw {np.asarray(x_raw).nbytes}B, "
                f"payload {len(payload)}B)."
            )
        return ratio
