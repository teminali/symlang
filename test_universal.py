#!/usr/bin/env python3
"""Test suite validating Universal Exact Codec on Code, Indentation, and Prompts."""

from symlang.universal_codec import UniversalExactCodec

PYTHON_SAMPLE = '''def calculate_metrics(transactions: list[dict], tax_rate: float = 0.08) -> dict:
    """Calculates financial totals with exact indentation."""
    subtotal = 0.0
    category_counts = {}
    
    for tx in transactions:
        amount = tx.get("amount", 0.0)
        if amount > 0:
            subtotal += amount
            cat = tx.get("category", "uncategorized")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
    tax = subtotal * tax_rate
    total = subtotal + tax
    
    return {
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "total": round(total, 2),
        "categories": category_counts,
    }'''

PROMPT_PARAGRAPHS = '''# System Prompt: Architectural Guidelines

You are an expert systems engineer. Adhere strictly to these principles:

1. Modularity & Decoupling:
   - Separate state from side effects.
   - Use dependency injection for IO boundaries.
   
2. Performance & Low Latency:
   - Avoid synchronous disk I/O in the event loop.
   - Profile memory allocations using jemalloc or tracemalloc.

Output your recommendation with bullet points, followed by executable code blocks.'''

JSON_SAMPLE = '''{
  "project": "SymLang",
  "version": "2.4.0",
  "settings": {
    "preserve_indentation": true,
    "max_recursion_depth": 1024,
    "allowed_encodings": ["utf-8", "ascii", "unicode_runes"]
  },
  "modules": [
    {"id": "meta_tier", "active": true},
    {"id": "universal_codec", "active": true}
  ]
}'''


def run_tests():
    print("=" * 105)
    print(" " * 24 + "UNIVERSAL CODEC TEST: CODE, INDENTATION & PROMPTS")
    print("=" * 105)
    
    samples = [
        ("Indented Python Code", PYTHON_SAMPLE),
        ("Multi-Paragraph Prompt", PROMPT_PARAGRAPHS),
        ("Structured JSON Payload", JSON_SAMPLE),
    ]
    
    print(f"{'Content Type':<26} | {'Lines':>5} | {'Original':>9} | {'Runes':>8} | {'Base85':>8} | {'Rune %':>8} | {'Fidelity'}")
    print("-" * 105)
    
    for name, content in samples:
        res = UniversalExactCodec.benchmark_arbitrary_content(content, name)
        status = "100% EXACT" if res["exact_fidelity_verified"] else "FAILED"
        print(
            f"{name:<26} | "
            f"{res['original_lines']:>5} | "
            f"{res['original_chars']:>7} c | "
            f"{res['rune_chars']:>6} c | "
            f"{res['b85_chars']:>6} c | "
            f"-{res['rune_reduction_pct']:>6.1f}% | "
            f"{status}"
        )
        assert res["exact_fidelity_verified"], f"Fidelity failed for {name}"
        
    print("-" * 105)
    print("\n[VERIFICATION OF INDENTATION & NEWLINES BYTE-FOR-BYTE]")
    for name, content in samples:
        runes = UniversalExactCodec.encode_to_runes(content)
        recovered = UniversalExactCodec.decode_from_runes(runes)
        assert recovered == content
        # Check specific whitespace properties
        assert "\n    for tx in transactions:" in recovered or "\n   - Separate state" in recovered or '\n    "preserve_indentation": true,' in recovered
        print(f"  ✓ {name}: All spaces, tabs, newlines, and quotes recovered identically.")
        
    print("\n" + "=" * 105)
    print(" " * 32 + "ALL TESTS PASSED: 100% LOSSLESS FIDELITY")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    run_tests()
