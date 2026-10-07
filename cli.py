#!/usr/bin/env python3
"""Command line for SymLang (alpha): lossy abbreviation tiers, one fixed grammar's meta-layers, and the token-aware router.

Scope, stated plainly:
* `encode` runs the pipeline's router and reports raw vs chosen tokens. Arbitrary text goes Direct (0% saving); the
  frame applies only to the one canonical sentence of the pre-agreed 294,912-sentence grammar. The lossy tiers it also
  prints have no decoder guarantee and can cost MORE tokens than the raw text (shown with a sign, never hidden).
* `meta-encode` / `meta-decode` work on states of that one grammar only. Input outside it is an error (exit code 2).
* Nothing here is a general text compressor and nothing here measures a real model.
"""

import argparse
import sys

from symlang import (
    SymbolicEncoder,
    SymbolicDecoder,
    LosslessSymbolicCodec,
)
from symlang.meta_tier import (
    Tier6Parser,
    MetaTierCodec,
    TOTAL_STATES,
)

EXIT_OK = 0
EXIT_BAD_INPUT = 2

SCOPE_NOTE = (
    "Scope: the frame/meta-layer results hold for ONE pre-agreed 294,912-sentence grammar (both sides must hold the "
    "same codebook). They are not general text compression; arbitrary text is sent Direct (0% saving)."
)


def _signed_pct(tokens: int, raw_tokens: int) -> str:
    """Signed token change vs raw: '-52.2%' fewer tokens, '+160.0%' MORE tokens, '0.0%' equal."""
    if raw_tokens <= 0:
        return "n/a"
    change = (tokens / raw_tokens - 1.0) * 100.0
    return "0.0%" if abs(change) < 0.05 else f"{change:+.1f}%"


def _print_router(pipeline, text: str) -> None:
    r = pipeline.prepare_llm_input(text)
    raw_tokens = r["original_tokens"]
    print(f"--- Router ({r['token_counter']} tokens; raw text is always a candidate) ---")
    print(f"Raw text:       {raw_tokens} tokens")
    print(f"Mode chosen:    {r['mode_used']}")
    print(f"Payload:        {r['compressed_tokens']} tokens ({r['input_token_saving_pct']:.1f}% saved; never negative)")
    print(f"Whole prompt:   {r['llm_prompt_tokens']} tokens incl. protocol line ({_signed_pct(r['llm_prompt_tokens'], raw_tokens)} tokens vs raw, one-shot)")
    if r["mode"] == "direct":
        print("No saving found: the text is sent as is.")
    print("Candidates considered (token change vs raw):")
    for c in r["candidates"]:
        flag = "eligible" if c["llm_readable"] else "not eligible for a prompt (opaque to an LLM)"
        verdict = "round trip ok" if c["roundtrip_verified"] else "round trip FAILED"
        print(f"  {c['key']:<15} {c['tokens']:>5} tok ({_signed_pct(c['tokens'], raw_tokens):>8})  {c['chars']:>5} chars  {verdict}; {flag}")
    if r["note"]:
        print(f"Note: {r['note']}")
    print()


def _print_tier(label: str, text: str, val: str, pipeline) -> None:
    raw_tokens = pipeline.count_tokens(text)
    toks = pipeline.count_tokens(val)
    print(f"--- {label} ---")
    print(f"Chars: {len(val)} ({_signed_pct(len(val), len(text))}) | Tokens: {toks} ({_signed_pct(toks, raw_tokens)} vs raw)")
    print(f"{val}\n")


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_BAD_INPUT


def _decode_meta(layer: str, text: str):
    """Decode a meta-layer string to a Tier6AST. Raises ValueError (with a clear message) on out-of-domain input."""
    if layer == "tensor":
        return MetaTierCodec.decode_layer7_tensor(text)
    if layer == "radix":
        return MetaTierCodec.decode_layer8_radix(text)
    if layer == "base85":
        return MetaTierCodec.decode_layer9a_base85(text)
    if layer == "rune":
        return MetaTierCodec.decode_layer9b_runes(text)
    if layer == "godel":
        try:
            number = int(text)
        except ValueError:
            raise ValueError(f"layer 'godel' expects a decimal integer, got {text!r}") from None
        return MetaTierCodec.decode_layer10_godel(number)
    if layer == "triad":
        if len(text) != 3:
            raise ValueError(f"layer 'triad' expects exactly 3 ASCII characters, got {len(text)} ({text!r})")
        # strict: raises ValueError for non-canonical characters (e.g. '!v!') and for ids outside [0, TOTAL_STATES)
        ast = MetaTierCodec.decode_layer11_triad(text)
        return ast
    if layer == "singular":
        if len(text) != 1:
            raise ValueError(
                f"layer 'singular' expects exactly 1 character (one Unicode code point), got {len(text)} ({text!r})"
            )
        return MetaTierCodec.decode_layer12_singular(text)  # raises ValueError for code points outside the grammar
    raise ValueError(f"unknown layer {layer!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SymLang (alpha): lossy abbreviation tiers, the meta-layers of one fixed grammar, and a token-aware router.",
        epilog=SCOPE_NOTE,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    encode_parser = subparsers.add_parser(
        "encode",
        help="Report the router's verdict (raw vs chosen tokens) and show the lossy abbreviation tiers",
    )
    encode_parser.add_argument("text", nargs="?", default="", help="Text to encode")
    encode_parser.add_argument(
        "--tier",
        choices=["all", "router", "telegraph", "logic", "nano", "unicode", "ultra", "glyph", "ast", "lossless"],
        default="all",
        help="What to show (default: all). 'router' shows only the token-aware router's verdict. "
        "Tiers other than 'lossless' are lossy and have no decoder guarantee.",
    )

    decode_parser = subparsers.add_parser("decode", help="Decode symbolic notation (lossy tiers: best effort only)")
    decode_parser.add_argument("symbolic_text", help="Symbolic text to decode")
    decode_parser.add_argument(
        "--mode",
        choices=["prompt", "deterministic", "lossless"],
        default="prompt",
        help="prompt: generate an LLM prompt, deterministic: expand rules, lossless: unpack the codebook glyphs",
    )

    meta_enc = subparsers.add_parser(
        "meta-encode", help="Encode a Tier 6 string (one fixed grammar) into the meta-layers"
    )
    meta_enc.add_argument("tier6_text", nargs="?", default="", help="Tier 6 symbolic string (must be in the grammar)")

    meta_dec = subparsers.add_parser(
        "meta-decode",
        help="Invert a meta-layer of the one fixed grammar back to Tier 6 (out-of-domain input is an error, exit 2)",
    )
    meta_dec.add_argument("meta_text", help="Meta-layer string (Tensor, Radix, Base85, Rune, Godel int, Triad or Singular)")
    meta_dec.add_argument(
        "--layer",
        choices=["tensor", "radix", "base85", "rune", "godel", "triad", "singular"],
        required=True,
        help="Layer of the input: tensor=7, radix=8, base85=9A, rune=9B, godel=10, "
        "triad=11 (3 ASCII chars '!'..'u'), singular=12 (1 code point U+10000..U+57FFF)",
    )

    proof_parser = subparsers.add_parser(
        "proof", help="Run the bijection check for the one fixed grammar (and print what it does not prove)"
    )
    proof_parser.add_argument(
        "--quick", action="store_true", help="check 1 state in 61 only (a sample, labelled as such) instead of all 294,912"
    )
    subparsers.add_parser("benchmark", help="Run benchmark on target sentence")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "proof":
        from proof import run_formal_proof
        run_formal_proof(stride=61 if args.quick else 1)
        return EXIT_OK

    if args.command == "benchmark" or not args.command:
        from benchmark import run_benchmark
        run_benchmark()
        return EXIT_OK

    encoder = SymbolicEncoder()
    decoder = SymbolicDecoder()

    if args.command == "encode":
        text = args.text
        if not text:
            return _error("please provide text to encode. Example: symlang encode 'While the model starts...'")

        from symlang.pipeline import NeuroSymbolicPipeline

        pipeline = NeuroSymbolicPipeline()
        print(f"\n[Original Text] ({len(text)} chars, {pipeline.count_tokens(text)} {pipeline.token_counter_name} tokens):\n{text}\n")

        if args.tier in ("all", "router"):
            _print_router(pipeline, text)
        if args.tier == "router":
            return EXIT_OK

        all_tiers = encoder.encode_all_tiers(text)
        tier_map = {
            "telegraph": ("Tier 1 (SymTelegraph, lossy)", all_tiers["tier1_telegraph"]),
            "logic": ("Tier 2 (SymLogic-ASCII, lossy)", all_tiers["tier2_symlogic_ascii"]),
            "nano": ("Tier 3 (SymNano Radix, lossy)", all_tiers["tier3_symnano"]),
            "unicode": ("Tier 4 (SymLogic-Unicode, lossy)", all_tiers["tier4_symlogic_unicode"]),
            "ultra": ("Tier 5 (SymOperator-Ultra, lossy)", all_tiers["tier5_operator_ultra"]),
            "glyph": ("Tier 6 (UltraGlyph, lossy)", all_tiers["tier6_ultraglyph"]),
            "ast": ("Tier 7 (SymAST, lossy)", all_tiers["tier7_symast"]),
        }

        def show_lossless() -> None:
            glyphs, raw = LosslessSymbolicCodec.encode_codebook_tokens(text)
            exact = LosslessSymbolicCodec.roundtrip_test(text)
            toks = pipeline.count_tokens(glyphs)
            raw_tokens = pipeline.count_tokens(text)
            print("--- Tier 8 (codebook glyphs) ---")
            print(
                f"Chars: {len(glyphs)} ({_signed_pct(len(glyphs), len(text))}) | Bytes: {len(raw)} | "
                f"Tokens: {toks} ({_signed_pct(toks, raw_tokens)} vs raw)"
            )
            print(f"Round trip for THIS input: {'exact' if exact else 'NOT exact (unexpected: the codec is meant to round-trip any text)'}")
            print(f"{glyphs}\n")

        if args.tier == "lossless":
            show_lossless()
            return EXIT_OK

        if args.tier == "all":
            print("Lossy abbreviation tiers follow: no decoder guarantee, and the token change can be positive (more tokens than raw).\n")
            for label, val in tier_map.values():
                _print_tier(label, text, val, pipeline)
            show_lossless()
        else:
            label, val = tier_map[args.tier]
            _print_tier(label, text, val, pipeline)
        return EXIT_OK

    if args.command == "decode":
        if args.mode == "lossless":
            try:
                expanded = LosslessSymbolicCodec.decode_unicode_glyphs(args.symbolic_text)
            except Exception as e:
                return _error(f"failed to decode lossless string: {e}")
            print(f"\n[Decoded English (codebook glyphs)]:\n{expanded}\n")
        elif args.mode == "deterministic":
            expanded = decoder.decode_deterministic(args.symbolic_text)
            print(f"\n[Deterministic Expanded English]:\n{expanded}\n")
        else:
            prompt = decoder.get_llm_decompression_prompt(args.symbolic_text)
            print("\n[Universal LLM Decompression Prompt]:")
            print("-" * 60)
            print(prompt)
            print("-" * 60)
        return EXIT_OK

    if args.command == "meta-encode":
        tier6_text = args.tier6_text or "▶M⇒PC~0 1-3m(1st:max).S:35-55G→RAM+GPU.✓ok.⏳¬✕win:win=👁S"
        try:
            ast = Tier6Parser.parse(tier6_text)
        except ValueError as e:
            return _error(str(e))
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
        print(f"Layer 10 (Gödel ℕ):    {l10} (Integer in ℕ)")
        try:
            l11 = MetaTierCodec.encode_layer11_triad(ast)
            l12 = MetaTierCodec.encode_layer12_singular(ast)
            print(f"Layer 11 (SymTriad):   {l11} ({len(l11)} chars, ASCII)")
            print(f"Layer 12 (SymSingular): {l12} (1 code point, U+{ord(l12):X})")
        except ValueError as e:
            print(f"Layer 11/12:           not available, {e}")
        print("\nChar counts are not token counts (run `symlang encode` or benchmark_tokens.py for tokens).")
        print(f"{SCOPE_NOTE}\n")
        return EXIT_OK

    if args.command == "meta-decode":
        try:
            ast = _decode_meta(args.layer, args.meta_text)
        except (ValueError, UnicodeError, IndexError) as e:
            return _error(f"cannot decode as layer '{args.layer}': {e}")
        except Exception as e:  # e.g. binascii.Error from a malformed Base85 string
            return _error(f"cannot decode as layer '{args.layer}': {type(e).__name__}: {e}")

        print(f"\n[Reconstructed Tier 6]:\n{ast.emit_tier6()}\n")
        try:
            print(f"(state {MetaTierCodec._ast_to_state_id(ast):,} of {TOTAL_STATES:,} in the one pre-agreed grammar)\n")
        except ValueError:
            print("(slot values outside the pre-agreed grammar's closed domains: no state id)\n")
        return EXIT_OK

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
