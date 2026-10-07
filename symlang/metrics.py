"""Metrics: compression ratios in characters, bytes and TOKENS, plus Shannon entropy.

Characters and bytes are not what an LLM bills or reads; tokens are, and they are tokenizer-specific. A representation
can lose 70% of its characters and still cost several times the tokens (see benchmark_tokens.py). Reductions here are
signed: a negative percentage means the "compressed" form is BIGGER, and it is reported as such.
"""

import math
from typing import Any, Callable, Dict, Tuple

try:
    import tiktoken
    ENC_CL100K = tiktoken.get_encoding("cl100k_base")
    ENC_O200K = tiktoken.get_encoding("o200k_base")
except Exception:
    ENC_CL100K = None
    ENC_O200K = None


def estimate_tokens(text: str) -> int:
    """Conservative token estimate used ONLY when tiktoken is unavailable (name: 'estimate').

    ASCII text is ~4 characters per token; every non-ASCII character is counted as 3 tokens (byte-fallback BPE spends
    2-4 tokens on rare code points, so rune-packed text is never under-counted into a fake saving).
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return -(-ascii_chars // 4) + 3 * (len(text) - ascii_chars)


def get_token_counter(encoding: str = "cl100k_base") -> Tuple[str, Callable[[str], int]]:
    """Return (name, counter) for a tiktoken encoding, or the conservative estimator if tiktoken cannot load it."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding)
        return f"tiktoken:{encoding}", lambda t: len(enc.encode(t, disallowed_special=()))
    except Exception:
        return "estimate:chars/4+3*non_ascii", estimate_tokens


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy (bits per character) of a text string."""
    if not text:
        return 0.0
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def evaluate_compression(original: str, compressed: str) -> Dict[str, Any]:
    """Calculate complete benchmark statistics comparing original and compressed text."""
    orig_chars = len(original)
    comp_chars = len(compressed)
    
    orig_bytes = len(original.encode("utf-8"))
    comp_bytes = len(compressed.encode("utf-8"))
    
    orig_words = len(original.split())
    comp_words = len(compressed.split())
    
    orig_cl100k = len(ENC_CL100K.encode(original)) if ENC_CL100K else 0
    comp_cl100k = len(ENC_CL100K.encode(compressed)) if ENC_CL100K else 0
    
    orig_o200k = len(ENC_O200K.encode(original)) if ENC_O200K else 0
    comp_o200k = len(ENC_O200K.encode(compressed)) if ENC_O200K else 0
    
    char_reduction = (1.0 - (comp_chars / orig_chars)) * 100.0 if orig_chars else 0.0
    byte_reduction = (1.0 - (comp_bytes / orig_bytes)) * 100.0 if orig_bytes else 0.0
    cl100k_reduction = (1.0 - (comp_cl100k / orig_cl100k)) * 100.0 if orig_cl100k else 0.0
    o200k_reduction = (1.0 - (comp_o200k / orig_o200k)) * 100.0 if orig_o200k else 0.0
    
    orig_entropy = calculate_entropy(original)
    comp_entropy = calculate_entropy(compressed)
    
    return {
        "original_chars": orig_chars,
        "compressed_chars": comp_chars,
        "char_reduction_pct": char_reduction,
        
        "original_bytes": orig_bytes,
        "compressed_bytes": comp_bytes,
        "byte_reduction_pct": byte_reduction,
        
        "original_words": orig_words,
        "compressed_words": comp_words,
        
        "original_cl100k": orig_cl100k,
        "compressed_cl100k": comp_cl100k,
        "cl100k_reduction_pct": cl100k_reduction,
        
        "original_o200k": orig_o200k,
        "compressed_o200k": comp_o200k,
        "o200k_reduction_pct": o200k_reduction,
        
        "original_entropy": orig_entropy,
        "compressed_entropy": comp_entropy,
    }
