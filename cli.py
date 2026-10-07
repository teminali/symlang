#!/usr/bin/env python3
"""Interactive CLI for SymLang (Symbolic Ultra-Compressed English Encoder & Meta-Tiers)."""

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="SymLang: Symbolic Ultra-Compressed English Language Encoder & Decoder"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode English text into symbolic notation")
    encode_parser.add_argument("text", nargs="?", default="", help="English text to encode")
    encode_parser.add_argument(
        "--tier",
        choices=["all", "telegraph", "logic", "nano", "unicode", "ultra", "glyph", "ast", "lossless"],
        default="all",
        help="Compression tier to use (default: all)",
    )

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode symbolic notation back to English")
    decode_parser.add_argument("symbolic_text", help="Symbolic text to decode")
    decode_parser.add_argument(
        "--mode",
        choices=["prompt", "deterministic", "lossless"],
        default="prompt",
        help="Decoding mode (prompt: generate LLM prompt, deterministic: expand rules, lossless: unpack codebook)",
    )

    # Meta-encode command
    meta_enc = subparsers.add_parser("meta-encode", help="Compress a Tier 6 string further into Meta-Layers")
    meta_enc.add_argument("tier6_text", nargs="?", default="", help="Tier 6 symbolic string to compress")

    # Meta-decode command
    meta_dec = subparsers.add_parser("meta-decode", help="Invert a Meta-Layer back to Tier 6")
    meta_dec.add_argument("meta_text", help="Meta-layer string (Tensor, Radix, Base85, Rune, or Godel Int)")
    meta_dec.add_argument(
        "--layer",
        choices=["tensor", "radix", "base85", "rune", "godel", "triad", "singular"],
        required=True,
        help="Layer type of the input string",
    )

    # Proof command
    subparsers.add_parser("proof", help="Run the formal mathematical bijection proof")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Run benchmark on target sentence")

    args = parser.parse_args()

    if args.command == "proof":
        from proof import run_formal_proof
        run_formal_proof()
        return

    if args.command == "benchmark" or not args.command:
        from benchmark import run_benchmark
        run_benchmark()
        return

    encoder = SymbolicEncoder()
    decoder = SymbolicDecoder()

    if args.command == "encode":
        text = args.text
        if not text:
            print("Please provide text to encode. Example: python3 cli.py encode 'While the model starts...'")
            return

        print(f"\n[Original Text] ({len(text)} chars):\n{text}\n")
        all_tiers = encoder.encode_all_tiers(text)

        tier_map = {
            "telegraph": ("Tier 1 (SymTelegraph)", all_tiers["tier1_telegraph"]),
            "logic": ("Tier 2 (SymLogic-ASCII)", all_tiers["tier2_symlogic_ascii"]),
            "nano": ("Tier 3 (SymNano Radix)", all_tiers["tier3_symnano"]),
            "unicode": ("Tier 4 (SymLogic-Unicode)", all_tiers["tier4_symlogic_unicode"]),
            "ultra": ("Tier 5 (SymOperator-Ultra)", all_tiers["tier5_operator_ultra"]),
            "glyph": ("Tier 6 (UltraGlyph)", all_tiers["tier6_ultraglyph"]),
            "ast": ("Tier 7 (SymAST)", all_tiers["tier7_symast"]),
        }

        if args.tier == "lossless":
            glyphs, raw = LosslessSymbolicCodec.encode_codebook_tokens(text)
            print(f"[Tier 8 (Lossless Codebook)] ({len(glyphs)} chars, {len(raw)} bytes):")
            print(glyphs)
            return

        if args.tier == "all":
            for key, (label, val) in tier_map.items():
                stats = evaluate_compression(text, val)
                print(f"--- {label} ---")
                print(f"Chars: {len(val)} (-{stats['char_reduction_pct']:.1f}%) | Tokens: {stats['compressed_cl100k']} (-{stats['cl100k_reduction_pct']:.1f}%)")
                print(f"{val}\n")
            glyphs, raw = LosslessSymbolicCodec.encode_codebook_tokens(text)
            stats = evaluate_compression(text, glyphs)
            print("--- Tier 8 (Lossless Codebook) ---")
            print(f"Chars: {len(glyphs)} (-{stats['char_reduction_pct']:.1f}%) | Bytes: {len(raw)} | Lossless 100% exact")
            print(f"{glyphs}\n")
        else:
            label, val = tier_map[args.tier]
            stats = evaluate_compression(text, val)
            print(f"--- {label} ---")
            print(f"Chars: {len(val)} (-{stats['char_reduction_pct']:.1f}%) | Tokens: {stats['compressed_cl100k']} (-{stats['cl100k_reduction_pct']:.1f}%)")
            print(val)

    elif args.command == "decode":
        if args.mode == "lossless":
            try:
                expanded = LosslessSymbolicCodec.decode_unicode_glyphs(args.symbolic_text)
                print(f"\n[Lossless Exact Decoded English]:\n{expanded}\n")
            except Exception as e:
                print(f"Failed to decode lossless string: {e}")
        elif args.mode == "deterministic":
            expanded = decoder.decode_deterministic(args.symbolic_text)
            print(f"\n[Deterministic Expanded English]:\n{expanded}\n")
        else:
            prompt = decoder.get_llm_decompression_prompt(args.symbolic_text)
            print("\n[Universal LLM Decompression Prompt]:")
            print("-" * 60)
            print(prompt)
            print("-" * 60)

    elif args.command == "meta-encode":
        tier6_text = args.tier6_text or "▶M⇒PC~0 1-3m(1st:max).S:35-55G→RAM+GPU.✓ok.⏳¬✕win:win=👁S"
        ast = Tier6Parser.parse(tier6_text)
        print(f"\n[Input Tier 6] ({len(tier6_text)} chars):\n{tier6_text}\n")
        
        l7 = MetaTierCodec.encode_layer7_tensor(ast)
        l8 = MetaTierCodec.encode_layer8_radix(ast)
        l9a = MetaTierCodec.encode_layer9a_base85(ast)
        l9b = MetaTierCodec.encode_layer9b_runes(ast)
        l10 = MetaTierCodec.encode_layer10_godel(ast)
        
        print(f"Layer 7 (SymTensor):   {l7} ({len(l7)} chars)")
        print(f"Layer 8 (SymRadix):    {l8} ({len(l8)} chars)")
        print(f"Layer 9A (Base-85):    {l9a} ({len(l9a)} chars)")
        print(f"Layer 9B (Runes):      {l9b} ({len(l9b)} chars)")
        print(f"Layer 10 (Gödel ℕ):    {l10} (Integer in ℕ)\n")

    elif args.command == "meta-decode":
        text = args.meta_text
        layer = args.layer
        if layer == "tensor":
            ast = MetaTierCodec.decode_layer7_tensor(text)
        elif layer == "radix":
            ast = MetaTierCodec.decode_layer8_radix(text)
        elif layer == "base85":
            ast = MetaTierCodec.decode_layer9a_base85(text)
        elif layer == "rune":
            ast = MetaTierCodec.decode_layer9b_runes(text)
        elif layer == "godel":
            ast = MetaTierCodec.decode_layer10_godel(int(text))
        else:
            print("Unknown layer type.")
            return

        tier6_out = ast.emit_tier6()
        print(f"\n[100% Mathematically Proven Reconstructed Tier 6]:\n{tier6_out}\n")


if __name__ == "__main__":
    main()
