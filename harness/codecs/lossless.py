"""Rung R0: lossless baseline codec (raw float32 bytes, ratio == 1.0)."""

import numpy as np

from harness.codecs.base import BaseCodec


class LosslessCodec(BaseCodec):
    """Bit-exact passthrough: payload is ``x.astype(float32).tobytes()``."""

    def encode(self, x: np.ndarray) -> bytes:
        """Return the raw little-endian float32 bytes of ``x``."""
        return np.ascontiguousarray(x, dtype=np.float32).tobytes()

    def decode(self, payload: bytes) -> np.ndarray:
        """Reconstruct the exact float32 array; raises on ragged length."""
        if len(payload) % 4 != 0:
            raise ValueError(f"R0 payload length {len(payload)} not a multiple of 4")
        return np.frombuffer(payload, dtype=np.float32).copy()
