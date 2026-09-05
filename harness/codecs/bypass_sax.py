"""Rung R4-bypass: SAX symbolic codec with an energy-bypass gate for spikes.

Why this exists: plain SAX saturates on spikes (nominal crests and +3.5-sigma
spikes both land in top bin 7, so D4 sees identical 7 -> 7 transitions and
Spike Event-F1 collapses to 0.00). This codec gates each PAA chunk on its raw
RMS energy ``E_k = sqrt(mean(x_k^2))``: nominal chunks encode as one SAX byte,
high-energy chunks emit a bypass record carrying the spike peak + offset.

BYTE FORMAT (little-endian throughout; framing is positional, never scanned):
  nominal chunk: ``[sym u8]`` with sym in [0, alphabet-1].
  flagged chunk: ``[0xFF][sym u8][peak f16LE][offset u8]`` (5 bytes).
``0xFF`` can never collide with a symbol (alphabet 8 -> symbols 0-7); f16 peak
bytes may contain 0xFF anywhere -- the parser consumes records positionally,
so embedded marker bytes are harmless (pinned by the 0xFF-collision test).
The stored ``sym`` is the chunk's SAX symbol, keeping the payload
self-contained for both ``decode`` (piecewise-constant base) and
``decode_tokens`` (base tokens + mask).

D4 IS UNCHANGED (8x8, ``direct_symbolic.py`` untouched): the hybrid lives in
the runner. This module freezes the margin constant and one pure helper:

RUNNER CONTRACT (Todo 9 owns the wiring; names below are STABLE):
  ``BYPASS_MARGIN = 0.5`` -- flagged test chunks score
  ``max-train-surprisal + BYPASS_MARGIN``. Suggested wiring, mirroring the
  existing R4 path in ``run_harness.run_cell`` (same floor(N/8)*8 truncation):
    1. ``codec.fit(xtr_tr)`` on the truncated nominal-train slice.
    2. ``toks_tr, mask_tr = codec.decode_tokens(codec.encode(xtr_tr))``.
    3. ``d4.fit(toks_tr[~mask_tr])`` -- D4 8x8 on nominal-only train tokens.
    4. ``toks_te, mask_te = codec.decode_tokens(codec.encode(xte_tr))``.
    5. ``tok = hybrid_token_surprisals(d4.transitions, toks_te, mask_te)``.
    6. ``scores = np.repeat(tok, 8)`` then ``_align_len`` (same as D4.score).
    7. Recommended tau: ``calibrate_frozen_threshold(train_scores, 100.0)``
       (MAX, not p99) over the train hybrid sample scores. Rationale: the
       margin puts flagged chunks 0.5 above the model's novelty ceiling, so
       max-tau suppresses every nominal score while keeping flagged chunks
       (detection == energy gate); p99-tau would admit novel-bigram FPs.
       Train-only, deterministic; equals the flagged value whenever train
       contains >= 1 flagged chunk (p99.5 gate => ~0.5% by construction).
  Bypass consumes raw floats only: ``fit``/``encode`` reject integer-dtype
  input, so bypass+Q2 stacking fails loudly instead of silently.
"""

import struct

import numpy as np
from scipy.stats import norm

from harness.codecs.base import BaseCodec
from harness.codecs.symbolic import BIN_MIDPOINTS, BREAKPOINTS

#: Frozen hybrid rule: flagged test chunks score max-train-surprisal + this.
#: Name and value are consumed by the Todo 9 runner wiring -- do not rename.
BYPASS_MARGIN = 0.5

#: Bypass record marker. Never a SAX symbol (alphabet 8 -> symbols 0-7).
MARKER = 0xFF

#: Smoothing twin of ``direct_symbolic.EPS``; nominal-equality with D4.score
#: is pinned by test (all-clear mask => helper == D4.score resampled).
_SURPRISAL_EPS = 1e-5

_FLAG_FMT = "<BBeB"  # marker, sym, peak-f16LE, offset
_FLAG_SIZE = struct.calcsize(_FLAG_FMT)


def _chunk_energies(x: np.ndarray, paa: int) -> np.ndarray:
    """Per-chunk RMS energies ``sqrt(mean(x_k^2))`` on raw (unnormalized) floats."""
    return np.sqrt(np.mean(np.square(x.reshape(-1, paa)), axis=1))


def hybrid_token_surprisals(
    transitions: np.ndarray,
    test_tokens: np.ndarray,
    test_mask: np.ndarray,
    margin: float = BYPASS_MARGIN,
) -> np.ndarray:
    """Per-chunk hybrid surprisals: D4 dynamics on nominal, margin rule on flagged.

    Nominal test chunks score exactly like ``DirectSymbolicDetector.score``
    (uniform ``-log(1/A + eps)`` at position 0, ``-log(P[t-1, t] + eps)``
    after); flagged chunks score ``MODEL-MAX + margin`` where MODEL-MAX is
    the maximum surprisal the TRAIN-fitted model can emit::

        max(-log(P[i, j] + eps)) over all cells (i, j)

    i.e. the novelty ceiling for unseen bigrams. The margin therefore
    guarantees flagged chunks STRICTLY dominate every score the model can
    produce on nominal input (nominal <= MODEL-MAX < MODEL-MAX + margin),
    so with ``tau = max(train hybrid scores)`` detection is exactly the
    energy gate. Pure function: numpy in, numpy out, no RNG, deterministic.

    Args:
        transitions: Fitted row-normalized AxA matrix (``d4.transitions``).
        test_tokens: Base int tokens per test chunk (from ``decode_tokens``).
        test_mask: Boolean bypass mask per test chunk (from ``decode_tokens``).
        margin: Surprisal margin for flagged chunks (default BYPASS_MARGIN).

    Returns:
        Float64 array of length ``len(test_tokens)`` (per-CHUNK; the runner
        resamples with ``np.repeat(scores, 8)`` exactly like ``D4.score``).

    Raises:
        ValueError: On mask/test length mismatch, float token dtype,
            out-of-range tokens, or a non-square transition matrix.
    """
    p = np.asarray(transitions, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] == 0:
        raise ValueError(f"transitions must be a non-empty square matrix, got {p.shape}.")
    n_states = p.shape[0]
    te = np.asarray(test_tokens)
    mk = np.asarray(test_mask, dtype=bool)
    if te.shape != mk.shape:
        raise ValueError(f"test/mask length mismatch: {te.shape} vs {mk.shape}.")
    if te.dtype.kind not in ("i", "u"):
        raise ValueError(f"test_tokens must be int tokens, got dtype {te.dtype}.")
    if te.size and (te.min() < 0 or te.max() >= n_states):
        raise ValueError(f"test_tokens must be in [0, {n_states - 1}].")
    te = te.astype(np.int64)
    model_max = max(float(np.max(-np.log(p + _SURPRISAL_EPS))),
                    -np.log(1.0 / n_states + _SURPRISAL_EPS))
    flagged_value = model_max + float(margin)
    out = np.empty(te.size, dtype=np.float64)
    if te.size:
        out[0] = -np.log(1.0 / n_states + _SURPRISAL_EPS)
        if te.size >= 2:
            out[1:] = -np.log(p[te[:-1], te[1:]] + _SURPRISAL_EPS)
    out[mk] = flagged_value
    return out


class BypassSAXCodec(BaseCodec):
    """SAX codec with a per-chunk RMS energy bypass gate for spike preservation."""

    def __init__(self, paa: int = 8, alphabet: int = 8) -> None:
        """Create an unfitted codec; call ``fit`` on the nominal train slice first."""
        self.paa = int(paa)
        self.alphabet = int(alphabet)
        if not 2 <= self.alphabet <= 255:
            raise ValueError(f"alphabet={alphabet} needs 2..255 (0xFF is the marker).")
        self.mean: float | None = None
        self.std: float | None = None
        self.theta: float | None = None

    def _breakpoints(self) -> np.ndarray:
        """Gaussian quantile cut points (frozen SAX tables for alphabet 8)."""
        if self.alphabet == 8:
            return BREAKPOINTS
        return norm.ppf(np.linspace(1 / self.alphabet, (self.alphabet - 1) / self.alphabet,
                                    self.alphabet - 1))

    def _midpoints(self) -> np.ndarray:
        """Bin midpoints for reconstruction (frozen SAX tables for alphabet 8)."""
        if self.alphabet == 8:
            return BIN_MIDPOINTS
        b = self._breakpoints()
        return np.concatenate([[(-3.0 + b[0]) / 2.0], (b[:-1] + b[1:]) / 2.0,
                               [(b[-1] + 3.0) / 2.0]])

    @staticmethod
    def _require_floats(x: np.ndarray, caller: str) -> np.ndarray:
        """Bypass consumes raw floats only; integer (quantized/token) input raises."""
        a = np.asarray(x)
        if a.dtype.kind in ("i", "u"):
            raise ValueError(
                f"BypassSAXCodec.{caller} takes raw float series only "
                f"(got dtype {a.dtype}); bypass+Q2 stacking is forbidden.")
        return a.astype(np.float64)

    def _check_len(self, n: int, caller: str) -> None:
        if n % self.paa != 0:
            raise ValueError(f"length {n} not divisible by paa={self.paa}.")

    def fit(self, x_train: np.ndarray) -> None:
        """Freeze train mean/std (SAX normalization) + energy gate theta.

        Theta is the 99.5th percentile of the train chunk RMS energies.
        Call on the R4-truncated nominal train slice (same rule as SAX fit);
        the flag rule is strict ``energy > theta``, so tied energies (e.g. a
        constant series) stay nominal.
        """
        x = self._require_floats(x_train, "fit")
        if x.size == 0:
            raise ValueError("BypassSAXCodec.fit needs a non-empty train slice.")
        self._check_len(x.size, "fit")
        self.mean = float(np.mean(x))
        self.std = float(np.std(x))
        if self.std == 0.0:
            self.std = 1e-12
        self.theta = float(np.percentile(_chunk_energies(x, self.paa), 99.5))

    def _symbols(self, x: np.ndarray) -> np.ndarray:
        """SAX symbols for chunked input under the frozen train stats."""
        z = (x - self.mean) / self.std  # type: ignore[operator]
        paa = z.reshape(-1, self.paa).mean(axis=1)
        return np.searchsorted(self._breakpoints(), paa).astype(np.uint8)

    def encode(self, x: np.ndarray) -> bytes:
        """Compress raw floats: 1 byte per nominal chunk, 5-byte record per flag."""
        if self.mean is None or self.std is None or self.theta is None:
            raise RuntimeError("BypassSAXCodec.encode called before fit.")
        x = self._require_floats(x, "encode")
        if x.size == 0:
            return b""
        self._check_len(x.size, "encode")
        syms = self._symbols(x)
        energies = _chunk_energies(x, self.paa)
        out = bytearray()
        for j in range(syms.size):
            if energies[j] <= self.theta:
                out.append(int(syms[j]))
            else:
                chunk = x[j * self.paa:(j + 1) * self.paa]
                off = int(np.argmax(np.abs(chunk)))
                out += struct.pack(_FLAG_FMT, MARKER, int(syms[j]),
                                   float(chunk[off]), off)
        return bytes(out)

    def _parse(self, payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Positional parse into (symbols, peaks, offsets, mask) per chunk."""
        buf = bytes(payload)
        syms: list[int] = []
        peaks: list[float] = []
        offs: list[int] = []
        masks: list[bool] = []
        i = 0
        while i < len(buf):
            if buf[i] == MARKER:
                if i + _FLAG_SIZE > len(buf):
                    raise ValueError(
                        f"truncated bypass record at byte {i} (need {_FLAG_SIZE}B).")
                _, sym, peak, off = struct.unpack_from(_FLAG_FMT, buf, i)
                if off >= self.paa:
                    raise ValueError(f"bypass offset {off} out of range for paa={self.paa}.")
                syms.append(sym)
                peaks.append(float(peak))
                offs.append(off)
                masks.append(True)
                i += _FLAG_SIZE
            else:
                syms.append(buf[i])
                peaks.append(0.0)
                offs.append(0)
                masks.append(False)
                i += 1
        return (np.array(syms, dtype=np.int64), np.array(peaks, dtype=np.float64),
                np.array(offs, dtype=np.int64), np.array(masks, dtype=bool))

    def decode(self, payload: bytes) -> np.ndarray:
        """Path A: piecewise-constant SAX reconstruction + peak impulses at offsets."""
        if self.mean is None or self.std is None:
            raise RuntimeError("BypassSAXCodec.decode called before fit.")
        syms, peaks, offs, masks = self._parse(payload)
        if syms.size == 0:
            return np.empty(0, dtype=np.float64)
        vals = self._midpoints()[syms] * self.std + self.mean
        rec = np.repeat(vals, self.paa).astype(np.float64)
        for j in np.flatnonzero(masks):
            rec[j * self.paa + offs[j]] = peaks[j]
        return rec

    def decode_tokens(self, payload: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Path B: (base int tokens, boolean bypass mask), one entry per chunk."""
        if self.mean is None or self.std is None or self.theta is None:
            raise RuntimeError("BypassSAXCodec.decode_tokens called before fit.")
        syms, _, _, masks = self._parse(payload)
        return syms, masks
