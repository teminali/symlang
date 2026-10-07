#!/usr/bin/env python3
"""Measured benchmark of the SymLang tiers and meta-layers on ONE sentence.

Everything printed is computed in this run: characters, UTF-8 bytes and tokens (counted by the pipeline's own token counter,
cl100k_base by default), the signed token change against the raw text (positive = more tokens than raw), and whether a decoder in this
repo really gives back the input text (checked, not asserted).

Scope: the frame rows (Tier 5 and the meta-layers) exist only for the ONE canonical sentence of a 294,912-sentence grammar.
They say nothing about other text. For any text outside the grammar pass --text: the frame rows are then skipped and said so.

    python benchmark.py                 # the canonical sentence
    python benchmark.py --text "..."    # any other text (frame rows skipped)
"""

import argparse
from typing import Callable, List, Optional, Tuple

from symlang import LosslessSymbolicCodec, SymbolicDecoder, SymbolicEncoder
from symlang.meta_tier import MetaTierCodec, Tier6Parser, verify_mathematical_roundtrip
from symlang.pipeline import NeuroSymbolicPipeline

TARGET_TEXT = (
    "While the model starts, your PC can be slow or stop responding for 1-3 minutes "
    "(longest the first time). Strata loads 35-55 GB into your RAM and locks part of "
    "it for the graphics card. This is normal. Wait, and don't close the window. "
    "The window shows what Strata is doing."
)

WIDTH = 108


def _signed_pct(new: int, old: int) -> str:
    """Signed change of `new` against `old` in percent: '-30.4%' is smaller, '+12.0%' is BIGGER than raw."""
    if not old:
        return "n/a"
    return f"{(new - old) / old * 100.0:+.1f}%"


def _recovers(text: str, form: str, decode: Optional[Callable[[str], str]]) -> str:
    """Run the decoder on the form and compare with the input: 'exact', 'NOT exact' or 'no decoder'."""
    if decode is None:
        return "no decoder"
    try:
        return "exact" if decode(form) == text else "NOT exact"
    except Exception as e:  # a decoder that raises did not recover the text
        return f"NOT exact ({type(e).__name__})"


def run_benchmark(text: str = TARGET_TEXT) -> None:
    pipeline = NeuroSymbolicPipeline()
    count = pipeline.count_tokens
    counter_name = pipeline.token_counter_name
    decoder = SymbolicDecoder()
    encoded = SymbolicEncoder().encode_all_tiers(text)

    # rows: (label, form, decoder or None). Decoders are real functions from this repo; "no decoder" is measured as such.
    rows: List[Tuple[str, str, Optional[Callable[[str], str]]]] = [
        ("Raw text", text, lambda s: s),
        ("Tier 1 SymTelegraph (lossy)", encoded["tier1_telegraph"], None),
        ("Tier 2 SymLogic-ASCII (lossy)", encoded["tier2_symlogic_ascii"], None),
        ("Tier 3 SymNano (lossy)", encoded["tier3_symnano"], None),
        ("Tier 4 SymLogic-Unicode (lossy)", encoded["tier4_symlogic_unicode"], None),
        ("Tier 7 SymAST (lossy)", encoded["tier7_symast"], None),
    ]

    # frame rows only when the text really is a sentence of the grammar
    tier6_str = encoded["tier5_operator_ultra"]
    ast = None
    frame_skipped_reason = ""
    try:
        ast = Tier6Parser.parse(tier6_str)
        MetaTierCodec.encode_layer11_triad(ast)  # raises if a slot value is outside the closed domain
        if decoder.decode_deterministic(tier6_str) != text:
            ast = None
            frame_skipped_reason = "the text is not a sentence of the 294,912-sentence grammar"
    except Exception as e:
        ast = None
        frame_skipped_reason = f"the text is not a sentence of the grammar ({type(e).__name__})"

    l7 = l8 = l9a = l9b = l10 = l11 = l12 = None
    if ast is not None:
        l7 = MetaTierCodec.encode_layer7_tensor(ast)
        l8 = MetaTierCodec.encode_layer8_radix(ast)
        l9a = MetaTierCodec.encode_layer9a_base85(ast)
        l9b = MetaTierCodec.encode_layer9b_runes(ast)
        l10 = MetaTierCodec.encode_layer10_godel(ast)
        l11 = MetaTierCodec.encode_layer11_triad(ast)
        l12 = MetaTierCodec.encode_layer12_singular(ast)

        def via(layer_decode: Callable[[object], object]) -> Callable[[str], str]:
            return lambda s: decoder.decode_deterministic(layer_decode(s).emit_tier6())

        rows += [
            ("Tier 5 SymOperator-Ultra [frame]", tier6_str, decoder.decode_deterministic),
            ("Layer 7 SymTensor [frame]", l7, decoder.decode_deterministic),
            ("Layer 8 SymRadix [frame]", l8, decoder.decode_deterministic),
            ("Layer 9A SymBase85 [frame]", l9a, via(MetaTierCodec.decode_layer9a_base85)),
            ("Layer 9B SymRune [frame]", l9b, via(MetaTierCodec.decode_layer9b_runes)),
            ("Layer 10 SymGodel [frame]", str(l10), via(lambda s: MetaTierCodec.decode_layer10_godel(int(s)))),
            ("Layer 11 SymTriad [frame]", l11, via(MetaTierCodec.decode_layer11_triad)),
            ("Layer 12 SymSingular [frame]", l12, via(MetaTierCodec.decode_layer12_singular)),
        ]
    glyphs, _ = LosslessSymbolicCodec.encode_codebook_tokens(text)
    rows.append(("Lossless codebook glyphs (any text)", glyphs, LosslessSymbolicCodec.decode_unicode_glyphs))

    raw_chars, raw_bytes, raw_tokens = len(text), len(text.encode("utf-8")), count(text)

    print("\n" + "=" * WIDTH)
    print(f" SymLang benchmark on ONE text | tokens: {counter_name} (the pipeline's counter) | every number measured in this run")
    print("=" * WIDTH)
    print(f"Text ({raw_chars} chars, {raw_tokens} tokens):\n\"{text}\"\n")
    print(f"{'Representation':<36} | {'Chars':>5} | {'Bytes':>5} | {'Tokens':>6} | {'Token chg':>9} | Decodes back")
    print("-" * WIDTH)
    for label, form, decode in rows:
        print(
            f"{label:<36} | {len(form):>5} | {len(form.encode('utf-8')):>5} | {count(form):>6} | "
            f"{_signed_pct(count(form), raw_tokens):>9} | {_recovers(text, form, decode)}"
        )
    print("-" * WIDTH)
    print("Token chg = token change against the raw text (negative = fewer tokens, positive = MORE). Chars and bytes are absolute. 'Decodes back' = a decoder in this repo was run and")
    print("compared with the input. 'no decoder' = the tier is lossy and nothing here reverses it.")

    if ast is None:
        print(f"\n[frame] rows skipped: {frame_skipped_reason}. They exist only for the one canonical sentence; this text would go Direct (0% saving).")
        print("=" * WIDTH + "\n")
        return

    print("\n[frame] rows hold for THIS sentence of ONE pre-agreed grammar only (11 slots, 294,912 sentences). Sender and receiver must share")
    print("the codebook: Layers 11/12 are short because the state lives in the codebook, not in the code. This is not general text compression.")

    print("\n" + "=" * WIDTH)
    print(" ROUND TRIP THROUGH THE META-LAYERS (this sentence only)")
    print("=" * WIDTH)
    proof = verify_mathematical_roundtrip(tier6_str)
    print(f"Tier 5 string:             {tier6_str} ({len(tier6_str)} chars)")
    print(f"-> Layer 7  (SymTensor):   {l7} ({len(l7)} chars)")
    print(f"-> Layer 8  (SymRadix):    {l8} ({len(l8)} chars)")
    print(f"-> Layer 9A (SymBase85):   {l9a} ({len(l9a)} chars)")
    print(f"-> Layer 9B (SymRune):     {l9b} ({len(l9b)} chars)")
    print(f"-> Layer 10 (SymGodel):    {l10}")
    print(f"-> Layer 11 (SymTriad):    {l11}")
    print(f"-> Layer 12 (SymSingular): {l12!r}")
    print("-" * WIDTH)
    recovered = MetaTierCodec.decode_layer9b_runes(l9b).emit_tier6()
    print(f"Layer 9B decoded back to the Tier 5 string:  identical = {recovered == tier6_str}")
    print(f"Layers 7, 8, 11, 12 each decode back to it:  identical = {proof['all_layers_mathematically_proven']}")
    print(f"Expanded to English equals the input text:   {decoder.decode_deterministic(recovered) == text}")
    print("(All-states check for the grammar: python proof.py. Nothing here covers text outside the grammar.)")
    print("=" * WIDTH + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measured SymLang benchmark on one text.")
    parser.add_argument("--text", default=TARGET_TEXT, help="text to measure (default: the canonical sentence of the grammar)")
    args = parser.parse_args(argv)
    run_benchmark(args.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
