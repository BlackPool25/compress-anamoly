"""Rung R3: uniform scalar quantization codecs (Q8, Q4, and Q2).

BYTE FORMAT FROZEN (little-endian throughout):
  Q8: 8-byte header [min float32 LE][max float32 LE] + N uint8 codes.
  Q4: same 8-byte header + ceil(N/2) bytes, nibble-packed HIGH-nibble-first;
      odd N pads the final low nibble with zero.
  Q2: same 8-byte header + ceil(N*2/8) bytes, 2-bit codes packed HIGH-first
      (first code in bits 7-6, then 5-4, 3-2, 1-0); N%4 != 0 pads the final
      byte's unused 2-bit slots with zero codes.
  min == max edge: all-zero codes; decode returns the constant value.

NOTE: Q4 payloads carry no length field, so ``decode`` of an odd-N payload
returns N+1 floats (trailing pad nibble decodes to the min value).
Q2 likewise carries no length field: ``decode`` of an N%4 != 0 payload
returns ceil(N/4)*4 floats (trailing pad codes decode to the min value);
the runner MUST pass every decode output through
``harness.run_harness._align_len`` which trims/pads to raw length N.

RATIO BAND FROZEN (measure-then-freeze, Task T5 evidence
.omo/evidence/compress-phase2-ladder/task-5-pass.log): Q2 [14.0, 17.0]
holds for synth N=1200 (4800 B raw -> 308 B payload = 15.58x) and UCR-scale
N=79795 (319180 B raw -> 19957 B payload = 15.99x). Ratio definition frozen:
ratio = x_raw.nbytes / len(payload).

RUNNER HANDOFF (Todo 9 owns harness/run_harness.py — do NOT wire Q2 there
in this change): wire Q2 Path-A by importing Q2Codec alongside Q8/Q4Codec
and adding an ``if "Q2" in codecs`` block mirroring the Q4 block
(encode -> path_a -> get_ratio); no other runner change needed since
_align_len already trims Q2 pad codes.
"""

import struct

import numpy as np

from harness.codecs.base import BaseCodec

_HEADER_FMT = "<ff"  # min, max as little-endian float32
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _header(mn: float, mx: float) -> bytes:
    """Pack the 8-byte [min, max] little-endian header."""
    return struct.pack(_HEADER_FMT, mn, mx)


def _unheader(payload: bytes) -> tuple:
    """Unpack the header; raises ValueError on truncated payloads."""
    if len(payload) < _HEADER_SIZE:
        raise ValueError(f"payload too short for header: {len(payload)} < {_HEADER_SIZE}")
    return struct.unpack(_HEADER_FMT, payload[:_HEADER_SIZE])


def _flat32(x: np.ndarray) -> np.ndarray:
    """Ravel input to a contiguous float32 vector."""
    return np.ascontiguousarray(x, dtype=np.float32).ravel()


class Q8Codec(BaseCodec):
    """R3a: uniform 8-bit quantization (256 levels)."""

    def encode(self, x: np.ndarray) -> bytes:
        """Quantize to uint8 codes behind the 8-byte header."""
        flat = _flat32(x)
        mn, mx = float(flat.min()), float(flat.max())
        if mx == mn:
            codes = np.zeros(flat.shape, dtype=np.uint8)
        else:
            scaled = (flat.astype(np.float64) - mn) / (mx - mn) * 255.0
            codes = np.rint(scaled).clip(0, 255).astype(np.uint8)
        return _header(mn, mx) + codes.tobytes()

    def decode(self, payload: bytes) -> np.ndarray:
        """Reconstruct float32 values from header + uint8 codes."""
        mn, mx = _unheader(payload)
        codes = np.frombuffer(payload, dtype=np.uint8, offset=_HEADER_SIZE).copy()
        if mx == mn:
            return np.full(codes.shape, mn, dtype=np.float32)
        return (mn + (mx - mn) * codes.astype(np.float64) / 255.0).astype(np.float32)


class Q4Codec(BaseCodec):
    """R3b: uniform 4-bit quantization (16 levels), nibble-packed."""

    def encode(self, x: np.ndarray) -> bytes:
        """Quantize to 4-bit codes, packed high-nibble-first (odd N zero-padded)."""
        flat = _flat32(x)
        mn, mx = float(flat.min()), float(flat.max())
        if mx == mn:
            codes = np.zeros(flat.shape, dtype=np.uint8)
        else:
            scaled = (flat.astype(np.float64) - mn) / (mx - mn) * 15.0
            codes = np.rint(scaled).clip(0, 15).astype(np.uint8)
        if codes.size % 2:
            codes = np.append(codes, np.uint8(0))  # ponytail: pad nibble, header has no length field
        packed = (codes[0::2] << 4) | codes[1::2]
        return _header(mn, mx) + packed.tobytes()

    def decode(self, payload: bytes) -> np.ndarray:
        """Unpack nibbles high-first and reconstruct float32 values."""
        mn, mx = _unheader(payload)
        raw = np.frombuffer(payload, dtype=np.uint8, offset=_HEADER_SIZE).copy()
        codes = np.empty(raw.size * 2, dtype=np.uint8)
        codes[0::2] = (raw >> 4) & 0xF
        codes[1::2] = raw & 0xF
        if mx == mn:
            return np.full(codes.shape, mn, dtype=np.float32)
        return (mn + (mx - mn) * codes.astype(np.float64) / 15.0).astype(np.float32)


class Q2Codec(BaseCodec):
    """R3c: uniform 2-bit quantization (4 levels), bit-packed high-first."""

    def encode(self, x: np.ndarray) -> bytes:
        """Quantize to 2-bit codes, packed high-first (N%4 != 0 zero-padded)."""
        flat = _flat32(x)
        mn, mx = float(flat.min()), float(flat.max())
        if mx == mn:
            codes = np.zeros(flat.shape, dtype=np.uint8)
        else:
            scaled = (flat.astype(np.float64) - mn) / (mx - mn) * 3.0
            codes = np.rint(scaled).clip(0, 3).astype(np.uint8)
        pad = (-codes.size) % 4
        if pad:
            codes = np.append(codes, np.zeros(pad, dtype=np.uint8))  # ponytail: pad codes, header has no length field
        packed = (codes[0::4] << 6) | (codes[1::4] << 4) | (codes[2::4] << 2) | codes[3::4]
        return _header(mn, mx) + packed.tobytes()

    def decode(self, payload: bytes) -> np.ndarray:
        """Unpack 2-bit codes high-first and reconstruct float32 values."""
        mn, mx = _unheader(payload)
        raw = np.frombuffer(payload, dtype=np.uint8, offset=_HEADER_SIZE).copy()
        codes = np.empty(raw.size * 4, dtype=np.uint8)
        codes[0::4] = (raw >> 6) & 0x3
        codes[1::4] = (raw >> 4) & 0x3
        codes[2::4] = (raw >> 2) & 0x3
        codes[3::4] = raw & 0x3
        if mx == mn:
            return np.full(codes.shape, mn, dtype=np.float32)
        return (mn + (mx - mn) * codes.astype(np.float64) / 3.0).astype(np.float32)
