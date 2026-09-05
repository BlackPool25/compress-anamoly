"""Temporal-convolutional autoencoder detector (Class 1b, w=64 stride 1).

Architecture (exact, frozen): Conv1D(1->16,k3,s2,p1)+ReLU ->
Conv1D(16->8,k3,s2,p1)+ReLU -> ConvTranspose1D(8->16,k3,s2,p1,op1)+ReLU ->
ConvTranspose1D(16->1,k3,s2,p1,op1). Total 905 params (< 12,000 asserted).

Window scores (per-window SSE in frozen train-normalized space) map to
points via TRAILING alignment; the first ``w - 1`` points use the median
of the TRAIN window scores, cached at ``fit`` time (same contract as
PCA/IF, so ``score`` is side-effect free).

TAU KEY (frozen, Todo 8 wires it): ``tau_TCN`` = p99 of
``TCNAutoencoder.score(R0-train)`` per (seed, dataset), reused across ALL
codecs, never recomputed per codec. Train-once-reuse across codecs is
Todo 9's job; this module delivers the single-train ``fit``/``score``
contract: ``fit`` on R0-train ONLY. Calibrate with
``harness.metrics.frozen_evaluator.calibrate_frozen_threshold`` (default
p99, mirroring PCA/IF) — NOT the D4 max-tau rule from Todo 6.

STAIRCASE-SCALE-SHIFT CAVEAT: coarse quantization (Q4/Q2) replaces smooth
nominal segments with piecewise-constant staircases. The TCN trained on
smooth R0-train reconstructs the smooth manifold, so quantization steps
inflate nominal window SSE vs R0 (Drift pilot: mean Q4 nominal residual >
mean R0 on all seeds 42-46). Under the frozen R0 tau this surfaces as
extra false positives on quantized codecs — the H2 mechanism, documented
not fixed: tau stays frozen, never retuned per codec.

TRAIN-WINDOW CAP (frozen subsample, Todo 8): UCR trains yield ~32k
windows, over budget. ``fit`` keeps at most ``MAX_TRAIN_WINDOWS`` windows
via :func:`frozen_subsample_indices` — ``step = ceil(n_win / 4000)``,
``offset = seed % step``, every ``step``-th window from ``offset`` — so
every worker builds identical trains from (n_win, seed) alone.

Determinism / versions (Context7-checked):
- Context7 library ID: ``/pytorch/pytorch`` (resolved 2026-09-05).
- Pinned ``torch==2.4.1`` CPU wheel (``2.4.1+cpu``) from the explicit
  ``pytorch-cpu`` uv index (https://download.pytorch.org/whl/cpu);
  ``uv.lock`` contains zero nvidia/cuda/triton entries.
- Verified APIs against Context7 docs: ``torch.manual_seed(seed)``,
  ``torch.use_deterministic_algorithms(True)`` (throws on nondeterministic
  ops), ``torch.set_num_threads(1)`` + ``torch.set_num_interop_threads(1)``,
  ``torch.backends.cudnn.deterministic=True`` /
  ``torch.backends.cudnn.benchmark=False`` (belt-and-braces; CPU-only).
- ``device='cpu'`` enforced; no CUDA path exists in this module.
- ``fit`` reseeds + rebuilds the net so refits are deterministic too.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import numpy as np

from harness.detectors.base import BaseDetector, sliding_windows
from harness.metrics.alignment import window_to_point_trailing

#: Frozen TCN window (4x the classical WINDOW; multiple of 4 so the
#: stride-2 x2 encoder/decoder round-trips exactly).
TCN_WINDOW: int = 64

#: Hard epoch cap (spec: <= 10; Todo 9's --fast profile uses 2).
MAX_EPOCHS: int = 10

#: Hard param ceiling, asserted at init.
MAX_PARAMS: int = 12_000

#: Score-time forward chunk size (Todo 10): ``_window_sse`` runs one batched
#: forward per chunk of at most this many windows. A single full-tensor
#: forward hits a slow oneDNN path (~4.6x slower on 60k windows); 1024-chunks
#: are bit-identical to it (max-abs-diff == 0.0, seeds 42/43). Chunking adds
#: no RNG consumption and no thread-config change (eval + no_grad as before).
SCORE_BATCH_WINDOWS: int = 1024

#: Frozen train-window cap (Todo 8): ``fit`` keeps at most this many
#: windows via :func:`frozen_subsample_indices`.
MAX_TRAIN_WINDOWS: int = 4000


def frozen_subsample_indices(
    n_win: int, seed: int, cap: int = MAX_TRAIN_WINDOWS
) -> np.ndarray:
    """Frozen window-subsample indices: identical trains on every worker.

    ``step = ceil(n_win / cap)``; ``offset = seed % step``; take every
    ``step``-th window starting at ``offset``. Asserts
    ``len == ceil((n_win - offset) / step)`` and ``len <= cap``.
    At or under cap this is the identity (``step == 1``, ``offset == 0``).

    Args:
        n_win: Total window count (``n - w + 1``).
        seed: Dataset seed; only ``seed % step`` enters the trains.
        cap: Window ceiling (frozen at 4000).

    Returns:
        1-D intp index array into the window axis.
    """
    import math

    n_win, seed, cap = int(n_win), int(seed), int(cap)
    step = math.ceil(n_win / cap)
    offset = seed % step
    idx = np.arange(offset, n_win, step)
    assert len(idx) == math.ceil((n_win - offset) / step)
    assert len(idx) <= cap
    return idx


#: torch.set_num_interop_threads may be called only once per process.
_INTEROP_THREADS_SET = False


def _apply_determinism(seed: int) -> None:
    """Seed + single-thread + deterministic-algorithms flags (CPU-only)."""
    global _INTEROP_THREADS_SET
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if not _INTEROP_THREADS_SET:
        torch.set_num_interop_threads(1)
        _INTEROP_THREADS_SET = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _TCNNet(nn.Module):
    """Exact frozen arch; 64 -> 32 -> 16 -> 32 -> 64."""

    def __init__(self) -> None:
        """Initialize frozen 1D TCN encoder-decoder network layers."""
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and transposed-conv decoder."""
        return self.dec(self.enc(x))


def _pad_to_mult4(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pad the last dim to a multiple of 4 (no-op for w=64 windows)."""
    n = t.shape[-1]
    pad = (-n) % 4
    if pad == 0:
        return t, 0
    return torch.nn.functional.pad(t, (0, pad)), pad


class TCNAutoencoder(BaseDetector):
    """TCN autoencoder on nominal windows; score = per-window SSE.

    Args:
        seed: Seeds init + training (byte-identical refits per seed).
        epochs: Full-batch Adam epochs; must be <= 10 (default 10).
        lr: Adam learning rate (default 1e-3).

    Raises:
        ValueError: If ``epochs`` > 10, inputs contain NaN/non-finite,
            or train data yields no windows.
        RuntimeError: If ``score`` is called before ``fit``.
    """

    def __init__(self, seed: int, epochs: int = 10,
                 lr: float = 1e-3) -> None:
        """Seed, enforce CPU determinism, build net, assert param ceiling."""
        if epochs > MAX_EPOCHS:
            raise ValueError(
                f"epochs={epochs} exceeds cap {MAX_EPOCHS}")
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.device = "cpu"
        _apply_determinism(seed)
        self._net = _TCNNet().to(self.device)
        n_params = self.param_count()
        assert n_params < MAX_PARAMS, (
            f"TCN params {n_params} >= ceiling {MAX_PARAMS}")
        self._mean: float = 0.0
        self._std: float = 1.0
        self._train_median: float = 0.0
        self._fitted = False

    def param_count(self) -> int:
        """Total trainable params (frozen arch: 905)."""
        return sum(p.numel() for p in self._net.parameters())

    @staticmethod
    def _as_clean_1d(x: np.ndarray) -> np.ndarray:
        """Ravel to float 1-D; raise ValueError on NaN/non-finite."""
        arr = np.asarray(x, dtype=float).ravel()
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("TCNAutoencoder input contains NaN/non-finite")
        return arr

    def _windows(self, arr: np.ndarray) -> torch.Tensor:
        """Frozen-normalized windows as a (n_win, 1, w) CPU tensor."""
        w = sliding_windows(arr, TCN_WINDOW)
        t = torch.from_numpy((w - self._mean) / self._std).float()
        return t[:, None, :].to(self.device)

    @torch.no_grad()
    def _window_sse(self, t: torch.Tensor) -> np.ndarray:
        """Per-window SSE of the frozen net (train-normalized space).

        Windows run in chunks of at most ``SCORE_BATCH_WINDOWS`` (one batched
        forward per chunk, concatenated); trailing alignment + train-median
        fill in :meth:`score` are untouched.
        """
        self._net.eval()
        parts = []
        for i in range(0, t.shape[0], SCORE_BATCH_WINDOWS):
            chunk = t[i : i + SCORE_BATCH_WINDOWS]
            recon, pad = _pad_to_mult4(self._net(_pad_to_mult4(chunk)[0]))
            if pad:
                recon = recon[..., : chunk.shape[-1]]
            parts.append(
                torch.sum((chunk - recon) ** 2, dim=(1, 2)).cpu().numpy())
        return np.concatenate(parts)

    def fit(self, x_train: np.ndarray) -> None:
        """Train once on R0-train windows; freeze norm + median fill."""
        arr = self._as_clean_1d(x_train)
        w = sliding_windows(arr, TCN_WINDOW)
        if w.shape[0] == 0:
            raise ValueError("train series too short for TCN window")
        self._mean = float(np.mean(arr))
        std = float(np.std(arr))
        self._std = std if std > 0 else 1.0
        # Reseed + rebuild so every fit from this seed is byte-identical.
        _apply_determinism(self.seed)
        self._net = _TCNNet().to(self.device)
        # Frozen subsample (Todo 8): identical capped trains per (n_win, seed).
        idx = frozen_subsample_indices(w.shape[0], self.seed)
        t = torch.from_numpy((w[idx] - self._mean) / self._std).float()
        t = t[:, None, :].to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            padded, pad = _pad_to_mult4(t)
            out = self._net(padded)
            if pad:
                out = out[..., : t.shape[-1]]
            loss = loss_fn(out, t)
            loss.backward()
            opt.step()
        self._train_median = float(np.median(self._window_sse(t)))
        self._fitted = True

    def score(self, x_test: np.ndarray) -> np.ndarray:
        """Window SSE mapped to points via trailing alignment."""
        if not self._fitted:
            raise RuntimeError("TCNAutoencoder.score called before fit")
        arr = self._as_clean_1d(x_test)
        n = int(arr.size)
        w = sliding_windows(arr, TCN_WINDOW)
        if w.shape[0] == 0:
            return np.full(n, self._train_median, dtype=float)
        ws = self._window_sse(self._windows(arr))
        return window_to_point_trailing(ws, n, TCN_WINDOW, self._train_median)
