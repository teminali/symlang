"""Lossless Exact Symbolic Codec for text.

Guarantee (tested, including a seeded property test over random Unicode): for EVERY Python ``str`` ``t``,
``decode_unicode_glyphs(encode_codebook_tokens(t)[0]) == t`` and ``decode_codebook_tokens(encode_codebook_tokens(t)[1]) == t``.
That covers non-ASCII text, emoji, combining marks, tabs, ``\\r\\n`` and even lone surrogates (any code point U+0000..U+10FFFF).

What it is: a byte-oriented codec with three kinds of token.
  * codebook phrase   one byte, the phrase index (0..len(SYSTEM_CODEBOOK)-1)
  * ASCII literal     one byte, ``0x80 | ord(c)``
  * escaped code point (any non-ASCII character)  4 bytes: the marker 0x7F then the code point as 3 big-endian bytes
Bytes are shown as one glyph each, ``chr(0x2400 + byte)``. ASCII input is encoded exactly as before this codec learned
non-ASCII; only non-ASCII characters use the escape.

What it is NOT: it is not a compressor for arbitrary text. Only the 14 fixed phrases in SYSTEM_CODEBOOK shrink (to one byte);
every other ASCII character costs one byte and every non-ASCII character costs four, so such text gets LARGER in bytes. The
glyph string also costs more LLM tokens than the raw text (see the README results). Use it for exact storage or transport.
"""

from typing import Tuple

# High-frequency grammatical n-grams in system guidance English
SYSTEM_CODEBOOK = [
    "While the model starts, ",
    "your PC can be slow ",
    "or stop responding for ",
    "1-3 minutes ",
    "(longest the first time). ",
    "Strata loads ",
    "35-55 GB into your RAM ",
    "and locks part of it for ",
    "the graphics card. ",
    "This is normal. ",
    "Wait, and don't close ",
    "the window. ",
    "The window shows what ",
    "Strata is doing.",
]


_ESCAPE = 0x7F  # unused codebook index; marks "the next 3 bytes are one code point"
_GLYPH_BASE = 0x2400


class LosslessSymbolicCodec:
    """Exact round-trip encoder and decoder: decode(encode(t)) == t for every str t (see the module docstring)."""

    @staticmethod
    def encode_codebook_tokens(text: str) -> Tuple[str, bytes]:
        """Encodes any str into (glyph string, packed bytes); phrases from SYSTEM_CODEBOOK become one byte."""
        packed = bytearray()
        i = 0
        n = len(text)
        while i < n:
            for idx, phrase in enumerate(SYSTEM_CODEBOOK):
                if text.startswith(phrase, i):
                    packed.append(idx)
                    i += len(phrase)
                    break
            else:
                cp = ord(text[i])
                if cp < 0x80:
                    packed.append(0x80 | cp)  # ASCII literal
                else:
                    packed.append(_ESCAPE)  # non-ASCII: marker + 3-byte big-endian code point (max U+10FFFF < 2**24)
                    packed += cp.to_bytes(3, "big")
                i += 1

        # Convert to compact Unicode glyph string (1 character per byte + offset to printable runes)
        unicode_glyphs = "".join(chr(_GLYPH_BASE + b) for b in packed)
        return unicode_glyphs, bytes(packed)

    @staticmethod
    def decode_codebook_tokens(packed_bytes: bytes) -> str:
        """Decodes packed bytes back into the exact original str. Raises ValueError on a malformed stream."""
        result = []
        i = 0
        n = len(packed_bytes)
        while i < n:
            b = packed_bytes[i]
            if b >= 0x80:
                result.append(chr(b & 0x7F))
                i += 1
            elif b == _ESCAPE:
                if i + 4 > n:
                    raise ValueError(f"truncated escape at byte {i}: need 3 bytes after the marker")
                cp = int.from_bytes(packed_bytes[i + 1 : i + 4], "big")
                if cp > 0x10FFFF:
                    raise ValueError(f"escaped code point {cp:#x} at byte {i} is past U+10FFFF")
                result.append(chr(cp))
                i += 4
            elif b < len(SYSTEM_CODEBOOK):
                result.append(SYSTEM_CODEBOOK[b])
                i += 1
            else:
                raise ValueError(f"byte {b:#x} at position {i} is not a codebook index, literal or escape")
        return "".join(result)

    @staticmethod
    def decode_unicode_glyphs(glyphs: str) -> str:
        """Decodes the glyph string back to the exact original text. Raises ValueError for a non-glyph character."""
        try:
            raw_bytes = bytes(ord(c) - _GLYPH_BASE for c in glyphs)
        except ValueError:
            raise ValueError(
                f"glyph string must only contain U+{_GLYPH_BASE:04X}..U+{_GLYPH_BASE + 0xFF:04X} characters"
            ) from None
        return LosslessSymbolicCodec.decode_codebook_tokens(raw_bytes)

    @staticmethod
    def roundtrip_test(text: str) -> bool:
        """Verify that encoding and decoding produces identical text."""
        glyphs, raw = LosslessSymbolicCodec.encode_codebook_tokens(text)
        decoded = LosslessSymbolicCodec.decode_unicode_glyphs(glyphs)
        return decoded == text
