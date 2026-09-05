"""Abstract base class for all compression codecs (rungs R0/R3/R4)."""

from abc import ABC, abstractmethod

import numpy as np


class BaseCodec(ABC):
    """Common interface: float array <-> raw bytes plus ratio helper."""

    @abstractmethod
    def encode(self, x: np.ndarray) -> bytes:
        """Compresses float array into raw bytes."""
        pass

    @abstractmethod
    def decode(self, payload: bytes) -> np.ndarray:
        """Decompresses raw bytes back into float array."""
        pass

    def get_ratio(self, x_raw: np.ndarray, payload: bytes) -> float:
        """Ratio of raw nbytes to payload length (higher = more compression)."""
        raw_bytes = x_raw.nbytes
        comp_bytes = len(payload)
        return raw_bytes / max(comp_bytes, 1)
