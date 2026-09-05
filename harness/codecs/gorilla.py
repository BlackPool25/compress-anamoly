"""Rung R1: exact Facebook-Gorilla XOR float compression (lossless).

BYTE FORMAT FROZEN (little-endian throughout):
  8-byte header [count u32LE][first f32LE raw bytes] + bitpacked XOR diffs.
  Bit order inside the body is MSB-first: the first control bit lands on
  bit 7 of byte 0; the final byte is zero-padded.

  For each value after the first, let ``xor = prev_bits ^ cur_bits``:
    ``0``                       -> value identical to previous.
    ``1`` + ``0``               -> xor nonzero, reuse previous (leading,
                                   trailing) window; followed by the
                                   ``32 - prev_leading - prev_trailing``
                                   meaningful bits (``xor >> prev_trailing``).
    ``1`` + ``1`` + 5-bit leading + 6-bit meaningful-length
      + meaningful bits         -> new window; meaningful length is
                                   ``32 - leading - trailing`` (1..32,
                                   stored raw; 32 fits in 6 bits).

  This is the Gorilla paper's control-bit scheme exactly (Pelkonen et al.,
  VLDB 2015, Fig. 4): zero-xor elision plus leading/trailing-zero window
  reuse. ``decode`` is bit-exact lossless: float32 words round-trip
  byte-identically, including NaN payloads (the header carries the raw
  float32 bytes of the first value, never a struct float round-trip).

RATIO BAND FROZEN (measured-then-frozen on pilot, 2026-09-05):
  GORILLA_RATIO_BAND = (0.90, 1.00); see evidence log
  ``.omo/evidence/compress-phase2-ladder/task-3-measure.log``.
  Pilot: 4 synth morphologies x seeds 42-46 -> 0.9405-0.9590; real UCR
  pinned-series test slice (N=47877) -> 0.9422; all bit-identical.
  Honest finding: Gorilla hovers at/below 1.0x on noisy float32 sensor
  data (mantissa noise kills leading/trailing-zero runs; per-diff control
  overhead meets or exceeds the bits saved). The initial [1.5, 3.5] guess
  (float64-regime expectation) is superseded by this measurement, never
  invented. R1's value is bit-exactness, not ratio.
  Ratio definition frozen repo-wide: ``x_raw.nbytes / len(payload)``.
"""

import struct

import numpy as np

from harness.codecs.base import BaseCodec

_COUNT_FMT = "<I"  # element count as little-endian uint32
_HEADER_SIZE = 8  # 4-byte count + 4-byte raw first float32

#: Measured-then-frozen ratio band (amended 2026-09-05 with pilot evidence).
GORILLA_RATIO_BAND: tuple[float, float] = (0.90, 1.00)


class _BitWriter:
    """MSB-first bit accumulator over a bytearray."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def write_bit(self, bit: int) -> None:
        """Append one bit (0/1)."""
        self.acc = (self.acc << 1) | (bit & 1)
        self.nbits += 1
        if self.nbits == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def write_bits(self, value: int, n: int) -> None:
        """Append the low ``n`` bits of value, MSB-first."""
        for i in range(n - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def finish(self) -> bytes:
        """Flush with zero padding; return body bytes."""
        if self.nbits:
            self.buf.append(self.acc << (8 - self.nbits))
        return bytes(self.buf)


class _BitReader:
    """MSB-first bit reader; overruns raise ValueError (never silent)."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.pos = 0  # bit position
        self.nbits = len(body) * 8

    def read_bit(self) -> int:
        """Read one bit; raises ValueError past the end."""
        if self.pos >= self.nbits:
            raise ValueError(
                f"gorilla body truncated at bit {self.pos} of {self.nbits}"
            )
        byte = self.body[self.pos // 8]
        bit = (byte >> (7 - (self.pos % 8))) & 1
        self.pos += 1
        return bit

    def read_bits(self, n: int) -> int:
        """Read ``n`` bits MSB-first as an int."""
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val

    def trailing_is_zero_padding(self) -> bool:
        """True iff every unread bit is zero (valid final-byte pad)."""
        p = self.pos
        while p < self.nbits:
            byte = self.body[p // 8]
            if (byte >> (7 - (p % 8))) & 1:
                return False
            p += 1
        return True


def _leading_zeros32(xor: int) -> int:
    """Leading zero bits of a nonzero 32-bit word."""
    return 32 - xor.bit_length()


def _trailing_zeros32(xor: int) -> int:
    """Trailing zero bits of a nonzero 32-bit word."""
    return ((xor & -xor).bit_length() - 1)


class GorillaCodec(BaseCodec):
    """R1: exact Gorilla XOR + leading/trailing bit-packing, lossless."""

    def encode(self, x: np.ndarray) -> bytes:
        """Pack float32 series behind the 8-byte LE header."""
        flat = np.ascontiguousarray(x, dtype=np.float32).ravel()
        n = int(flat.size)
        if n == 0:
            raise ValueError("gorilla encode refuses empty input")
        words = flat.view(np.uint32)
        header = struct.pack(_COUNT_FMT, n) + flat[:1].tobytes()
        if n == 1:
            return header
        w = _BitWriter()
        prev = int(words[0])
        prev_lead, prev_trail = 0, 0
        have_window = False
        for i in range(1, n):
            cur = int(words[i])
            xor = prev ^ cur
            if xor == 0:
                w.write_bit(0)
            else:
                w.write_bit(1)
                lead = _leading_zeros32(xor)
                trail = _trailing_zeros32(xor)
                if have_window and lead >= prev_lead and trail >= prev_trail:
                    w.write_bit(0)
                    w.write_bits(xor >> prev_trail, 32 - prev_lead - prev_trail)
                else:
                    w.write_bit(1)
                    meaningful = 32 - lead - trail
                    w.write_bits(lead, 5)
                    w.write_bits(meaningful, 6)
                    w.write_bits(xor >> trail, meaningful)
                    prev_lead, prev_trail = lead, trail
                    have_window = True
            prev = cur
        return header + w.finish()

    def decode(self, payload: bytes) -> np.ndarray:
        """Reconstruct the exact float32 array; raises on truncation."""
        if len(payload) < _HEADER_SIZE:
            raise ValueError(
                f"gorilla payload too short for header: {len(payload)} < {_HEADER_SIZE}"
            )
        (count,) = struct.unpack(_COUNT_FMT, payload[:4])
        if count <= 0:
            raise ValueError(f"gorilla payload has invalid count {count}")
        first = np.frombuffer(payload[4:8], dtype=np.float32).copy()
        out = np.empty(count, dtype=np.float32)
        out[0] = first[0]
        if count == 1:
            if len(payload) != _HEADER_SIZE or not _BitReader(payload[8:]).trailing_is_zero_padding():
                raise ValueError("gorilla count-1 payload carries trailing garbage")
            return out
        if len(payload) == _HEADER_SIZE:
            raise ValueError("gorilla payload missing diff body")
        r = _BitReader(payload[_HEADER_SIZE:])
        words = np.empty(count, dtype=np.uint32)
        words[0] = out.view(np.uint32)[0]
        prev = int(words[0])
        prev_lead, prev_trail = 0, 0
        have_window = False
        for i in range(1, count):
            if r.read_bit() == 0:
                cur = prev
            else:
                if r.read_bit() == 0:
                    if not have_window:
                        raise ValueError(
                            f"gorilla corrupt window-reuse at index {i}"
                        )
                    cur = prev ^ (r.read_bits(32 - prev_lead - prev_trail) << prev_trail)
                else:
                    lead = r.read_bits(5)
                    meaningful = r.read_bits(6)
                    if not (1 <= meaningful <= 32 and 0 <= lead <= 31):
                        raise ValueError(
                            f"gorilla corrupt window header at index {i}: "
                            f"lead={lead} meaningful={meaningful}"
                        )
                    trail = 32 - lead - meaningful
                    if trail < 0:
                        raise ValueError(
                            f"gorilla corrupt window geometry at index {i}: "
                            f"lead={lead} meaningful={meaningful}"
                        )
                    cur = prev ^ (r.read_bits(meaningful) << trail)
                    prev_lead, prev_trail = lead, trail
                    have_window = True
            words[i] = np.uint32(cur)
            prev = cur
        leftover = r.nbits - r.pos
        if leftover >= 8 or not r.trailing_is_zero_padding():
            raise ValueError("gorilla payload carries trailing garbage bits")
        return words.view(np.float32)
