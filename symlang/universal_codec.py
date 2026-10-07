"""Generic lossless codec: UTF-8 -> zlib (DEFLATE) -> Unicode runes or Base85.

What it is: an exact round trip for arbitrary text (newlines, indentation, code, emoji all survive byte-for-byte,
verified by tests). Savings are in CHARACTERS: typically 60-85% fewer characters on code and prose of a few hundred
characters or more.

What it is not: a token saver. The output is DEFLATE bytes shown as code points or Base85 text, which BPE tokenizers
split into many tokens, so it costs about 1.8-4x MORE tokens than the raw text (measured in cl100k_base and o200k_base;
run `python benchmark_tokens.py`). It is also opaque: an LLM cannot read or inflate it. Use it for storage and transport
between your own systems, never as a prompt compression. Short inputs may not shrink at all (zlib header + checksum
cost 8 bytes).

Rune layout: `encode_to_runes` packs two DEFLATE bytes per character into Unicode plane 1 (U+10000..U+1FFFF); an odd
stream is padded with one 0xFF byte that the decoder strips.
"""

import zlib
import base64
from typing import Tuple, Dict, Any


class UniversalExactCodec:
    """Universal lossless codec for arbitrary code, indentation, and text."""

    @staticmethod
    def encode_to_runes(text: str, level: int = 9) -> str:
        """Packs arbitrary text/code into 16-bit Unicode Runes preserving exact whitespace.
        
        Uses DEFLATE bitstream folded into Unicode Plane 1 (U+10000..U+1FFFF, surrogate-free).
        Every character carries 2 packed bytes.
        """
        raw_bytes = text.encode("utf-8")
        compressed = zlib.compress(raw_bytes, level=level)
        padded = bytearray(compressed)
        
        # Ensure even length
        if len(padded) % 2 != 0:
            padded.append(0xFF)  # Sentinel padding marker
            
        runes = []
        for i in range(0, len(padded), 2):
            val = (padded[i] << 8) | padded[i + 1]
            runes.append(chr(0x10000 + val))
            
        return "".join(runes)

    @staticmethod
    def decode_from_runes(rune_str: str) -> str:
        """Decodes 16-bit Unicode Runes back into the exact original text/code."""
        unpacked = bytearray()
        for r in rune_str:
            val = ord(r) - 0x10000
            unpacked.append((val >> 8) & 0xFF)
            low = val & 0xFF
            unpacked.append(low)
            
        # Try decompressing directly or stripping trailing padding if needed
        try:
            raw_bytes = zlib.decompress(bytes(unpacked))
        except Exception:
            # Strip trailing sentinel if present
            if unpacked and unpacked[-1] == 0xFF:
                unpacked = unpacked[:-1]
            raw_bytes = zlib.decompress(bytes(unpacked))
            
        return raw_bytes.decode("utf-8")

    @staticmethod
    def encode_to_base85(text: str, level: int = 9) -> str:
        """Packs arbitrary text/code into clean printable 7-bit ASCII Base-85."""
        raw_bytes = text.encode("utf-8")
        compressed = zlib.compress(raw_bytes, level=level)
        return "§" + base64.b85encode(compressed).decode("ascii")

    @staticmethod
    def decode_from_base85(b85_str: str) -> str:
        """Decodes Base-85 back into exact original text/code."""
        if b85_str.startswith("§"):
            b85_str = b85_str[1:]
        raw_bytes = base64.b85decode(b85_str.encode("ascii"))
        return zlib.decompress(raw_bytes).decode("utf-8")

    @staticmethod
    def benchmark_arbitrary_content(content: str, name: str = "Sample") -> Dict[str, Any]:
        """Round-trips `content` through both codecs and reports CHARACTER reductions (not token counts; see benchmark_tokens.py)."""
        rune_repr = UniversalExactCodec.encode_to_runes(content)
        recovered_rune = UniversalExactCodec.decode_from_runes(rune_repr)
        
        b85_repr = UniversalExactCodec.encode_to_base85(content)
        recovered_b85 = UniversalExactCodec.decode_from_base85(b85_repr)
        
        orig_chars = len(content)
        orig_bytes = len(content.encode("utf-8"))
        rune_chars = len(rune_repr)
        b85_chars = len(b85_repr)
        
        char_reduction_rune = (1.0 - (rune_chars / orig_chars)) * 100.0 if orig_chars else 0.0
        char_reduction_b85 = (1.0 - (b85_chars / orig_chars)) * 100.0 if orig_chars else 0.0
        
        exact_fidelity = (recovered_rune == content) and (recovered_b85 == content)
        
        return {
            "name": name,
            "original_chars": orig_chars,
            "original_bytes": orig_bytes,
            "original_lines": len(content.splitlines()),
            "rune_chars": rune_chars,
            "rune_reduction_pct": char_reduction_rune,
            "b85_chars": b85_chars,
            "b85_reduction_pct": char_reduction_b85,
            "exact_fidelity_verified": exact_fidelity,
            "rune_sample": rune_repr[:30] + ("..." if len(rune_repr) > 30 else ""),
        }
