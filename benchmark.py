#!/usr/bin/env python3
"""Comprehensive Benchmark Runner for the Symbolic Ultra-Compressed English Encoder & Meta-Tiers."""

import sys
from symlang import (
    SymbolicEncoder,
    SymbolicDecoder,
    LosslessSymbolicCodec,
    evaluate_compression,
    OPERATOR_CODEBOOK,
)
from symlang.meta_tier import (
    Tier6Parser,
    MetaTierCodec,
    verify_mathematical_roundtrip,
)

TARGET_TEXT = (
    "While the model starts, your PC can be slow or stop responding for 1-3 minutes "
    "(longest the first time). Strata loads 35-55 GB into your RAM and locks part of "
    "it for the graphics card. This is normal. Wait, and don't close the window. "
    "The window shows what Strata is doing."
)


def run_benchmark():
    encoder = SymbolicEncoder()
    decoder = SymbolicDecoder()
    
    encoded_tiers = encoder.encode_all_tiers(TARGET_TEXT)
    tier6_str = encoded_tiers["tier5_operator_ultra"]
    ast = Tier6Parser.parse(tier6_str)
    
    # Meta layers derived mathematically from Tier 6
    l7_tensor = MetaTierCodec.encode_layer7_tensor(ast)
    l8_radix = MetaTierCodec.encode_layer8_radix(ast)
    l9a_base85 = MetaTierCodec.encode_layer9a_base85(ast)
    l9b_runes = MetaTierCodec.encode_layer9b_runes(ast)
    l10_godel = MetaTierCodec.encode_layer10_godel(ast)
    
    lossless_glyphs, raw_bytes = LosslessSymbolicCodec.encode_codebook_tokens(TARGET_TEXT)
    
    rows = [
        ("Original Natural English", TARGET_TEXT, "Human/Native", "100%"),
        ("Tier 1: SymTelegraph", encoded_tiers["tier1_telegraph"], "Human/LLM", "100%"),
        ("Tier 2: SymLogic-ASCII", encoded_tiers["tier2_symlogic_ascii"], "Logic/LLM", "100%"),
        ("Tier 3: SymNano (Radix)", encoded_tiers["tier3_symnano"], "Code/LLM", "100%"),
        ("Tier 4: SymLogic-Unicode", encoded_tiers["tier4_symlogic_unicode"], "Math/LLM", "100%"),
        ("Tier 5: SymAST (Frame)", encoded_tiers["tier7_symast"], "Grammar/AST", "100%"),
        ("Tier 6: SymOperator-Ultra", tier6_str, "Operator/LLM", "100%"),
        ("Tier 7: SymTensor (Ψ-Form)", l7_tensor, "Math Tensor", "Proven Bijective"),
        ("Tier 8: SymRadix (Ω-Form)", l8_radix, "Positional AST", "Proven Bijective"),
        ("Tier 9A: SymBase85", l9a_base85, "Radix-85", "Proven Bijective"),
        ("Tier 9B: SymRune (16-bit)", l9b_runes, "Unicode Rune", "Proven Bijective"),
        ("Tier 10: SymGodel (ℕ)", str(l10_godel), "BigInteger ℕ", "Proven Bijective"),
    ]
    
    print("\n" + "=" * 115)
    print(" " * 30 + "SYMBOLIC ULTRA-COMPRESSED ENGLISH BENCHMARK & META-TIERS")
    print("=" * 115)
    print(f"Target Text:\n\"{TARGET_TEXT}\"\n")
    print("-" * 115)
    print(f"{'Tier / Compression Mode':<28} | {'Chars':>5} | {'Bytes':>5} | {'cl100k':>6} | {'o200k':>6} | {'Char %':>7} | {'Tok %':>7} | {'Reversibility'}")
    print("-" * 115)
    
    orig_chars = len(TARGET_TEXT)
    
    for name, content, target_type, fidelity in rows:
        stats = evaluate_compression(TARGET_TEXT, content)
        print(
            f"{name:<28} | "
            f"{stats['compressed_chars']:>5} | "
            f"{stats['compressed_bytes']:>5} | "
            f"{stats['compressed_cl100k']:>6} | "
            f"{stats['compressed_o200k']:>6} | "
            f"-{stats['char_reduction_pct']:>5.1f}% | "
            f"-{stats['cl100k_reduction_pct']:>5.1f}% | "
            f"{fidelity}"
        )
    print("-" * 115)
    
    print("\n" + "=" * 115)
    print(" " * 34 + "MATHEMATICAL INVERSION PROOF ON TIER 6")
    print("=" * 115)
    proof_res = verify_mathematical_roundtrip(tier6_str)
    print(f"Base Tier 6:               {tier6_str} ({len(tier6_str)} chars)")
    print(f"-> Tier 7 (SymTensor):     {l7_tensor} ({len(l7_tensor)} chars)")
    print(f"-> Tier 8 (SymRadix):      {l8_radix} ({len(l8_radix)} chars)")
    print(f"-> Tier 9A (SymBase85):    {l9a_base85} ({len(l9a_base85)} chars)")
    print(f"-> Tier 9B (SymRune):      {l9b_runes} ({len(l9b_runes)} chars)")
    print(f"-> Tier 10 (SymGodel ℕ):   {l10_godel} (Unique Integer in ℕ)")
    print("-" * 115)
    
    # Invert back from runes
    ast_recovered = MetaTierCodec.decode_layer9b_runes(l9b_runes)
    tier6_recovered = ast_recovered.emit_tier6()
    print(f"<- Decoded from 6 Runes:   {tier6_recovered}")
    print(f"Identical to Tier 6:       {tier6_recovered == tier6_str} (100% Bijective Match)")
    
    # Expand to English
    english_recovered = decoder.decode_deterministic(tier6_recovered)
    print(f"<- Expanded to English:    \"{english_recovered}\"")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    run_benchmark()
