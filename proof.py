#!/usr/bin/env python3
"""Exhaustive bijection check for ONE pre-agreed grammar, and an explicit statement of what it does not prove.

What this script verifies (by enumerating every state, not by sampling):

1. Entropy of the grammar: 11 slots with small closed domains give |Omega| = 294,912 states = log2 = 18.17 bits.
2. AST <-> state id <-> Layer 8 (radix) / Layer 11 (3 ASCII chars) / Layer 12 (1 code point) over ALL 294,912 states:
   every state has a distinct code in each layer and decodes back to itself.
3. Tier 6 string <-> AST over all states (P(E(a)) = a and distinct strings), and sentence <-> AST over all states
   (E(P(s)) = s for every sentence of the frame, every state expands to a distinct sentence).
4. Out-of-domain input (a state id >= 294,912, a slot value outside its domain) is rejected with ValueError.

What it does NOT prove is printed at the end. In short: nothing about arbitrary text, nothing about token cost, nothing
about any model. The compact codes exist only because both sides hold the same codebook.
"""

import math
import time

from symlang.decoder import expand_frame, parse_canonical_sentence
from symlang.meta_tier import (
    DOMAIN_SLOTS,
    TOTAL_STATES,
    MetaTierCodec,
    Tier6Parser,
)
from symlang.metrics import get_token_counter

SAMPLE_TIER6 = "▶M⇒PC~0 1-3m(1st:max).S:35-55G→RAM+GPU.✓ok.⏳¬✕win:win=👁S"

SCOPE_DISCLAIMER = """\
WHAT THIS DOES NOT PROVE (scope of the result)
  * It holds for ONE pre-agreed grammar of 294,912 sentences. Text outside it is not parsed, not guessed and not
    decoded; the pipeline sends it Direct (0% saving). It says nothing about arbitrary text or code.
  * It says nothing about token cost. The 3-char and 1-glyph codes are short in characters, but a 1-glyph code costs
    several tokens (4 in cl100k_base here, not 1), and a receiver without the codebook cannot read them at all.
  * The generic lossless codecs (zlib+runes, zlib+base85) are exact but cost MORE LLM tokens than the raw text: measured
    1.8-3.3x (runes) and 0.9-1.6x (base85) on three files, see README.md and benchmark_tokens.py. Use them for storage
    or transport, not prompt compression.
  * No real model is involved: nothing here measures a model's behaviour, latency or answer quality.
"""


def _exhaustive_check(stride: int = 1):
    """Walk every `stride`-th state id (stride 1 = all of them); returns (checked, failures: list of str)."""
    failures = []
    tier6_seen = set()
    sentence_seen = set()
    triad_seen = set()
    glyph_seen = set()
    radix_seen = set()
    state_ids = range(0, TOTAL_STATES, stride)
    for state_id in state_ids:
        ast = MetaTierCodec._state_id_to_ast(state_id)

        t6 = ast.emit_tier6()
        tier6_seen.add(t6)
        if Tier6Parser.parse(t6) != ast:
            failures.append(f"Tier 6 parse(emit(a)) != a at state {state_id}")

        sentence = expand_frame(ast)
        sentence_seen.add(sentence)
        if parse_canonical_sentence(sentence) != ast:
            failures.append(f"sentence parse(expand(a)) != a at state {state_id}")

        radix = MetaTierCodec.encode_layer8_radix(ast)
        radix_seen.add(radix)
        if MetaTierCodec.decode_layer8_radix(radix) != ast:
            failures.append(f"Layer 8 round trip failed at state {state_id}")

        triad = MetaTierCodec.encode_layer11_triad(ast)
        triad_seen.add(triad)
        if MetaTierCodec.decode_layer11_triad(triad) != ast:
            failures.append(f"Layer 11 round trip failed at state {state_id}")

        glyph = MetaTierCodec.encode_layer12_singular(ast)
        glyph_seen.add(glyph)
        if MetaTierCodec.decode_layer12_singular(glyph) != ast:
            failures.append(f"Layer 12 round trip failed at state {state_id}")

        if len(failures) > 10:
            break

    for name, seen in (
        ("Tier 6 strings", tier6_seen),
        ("sentences", sentence_seen),
        ("Layer 8 codes", radix_seen),
        ("Layer 11 codes", triad_seen),
        ("Layer 12 codes", glyph_seen),
    ):
        if len(seen) != len(state_ids):
            failures.append(f"{name}: {len(seen):,} distinct for {len(state_ids):,} states (not injective)")
    return len(state_ids), failures


def _rejects(fn, *args) -> bool:
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def run_formal_proof(stride: int = 1) -> bool:
    """Run the checks and print the report. Returns True if every check passed (raises AssertionError otherwise).

    stride=1 (default) enumerates all 294,912 states. A larger stride checks every stride-th state and says so in the
    report (a sample, not a proof); it exists so tests and quick runs do not pay for the full enumeration.
    """
    print("=" * 105)
    kind = "EXHAUSTIVE" if stride == 1 else f"SAMPLED (1 state in {stride}, NOT exhaustive)"
    print(" " * 6 + f"{kind} BIJECTION CHECK FOR ONE PRE-AGREED GRAMMAR (294,912 STATES)")
    print("=" * 105)

    total_states = 1
    for _slot_name, values in DOMAIN_SLOTS:
        total_states *= len(values)
    assert total_states == TOTAL_STATES
    entropy = math.log2(total_states)

    print("\n[STEP 1: STATE SPACE AND ENTROPY OF THE GRAMMAR]")
    print(f"Slots: {' x '.join(str(len(v)) for _, v in DOMAIN_SLOTS)} = {total_states:,} sentences")
    print(f"Entropy of the grammar:                     log2({total_states:,}) = {entropy:.4f} bits")
    print(f"Minimum whole bytes to index a state:       {math.ceil(entropy / 8)}")
    print(f"Capacity check, Base-85 triad (85^3):       {85**3:,} >= {total_states:,}")
    print(f"Capacity check, Unicode code points:        {0x110000:,} >= {total_states:,}")
    print("These bounds count states of THIS grammar; they are not a bound on compressing text in general.")

    print("\n[STEP 2: " + ("EXHAUSTIVE ENUMERATION OF ALL STATES" if stride == 1 else "SAMPLE OF STATES (NOT EXHAUSTIVE)") + "]")
    start = time.time()
    checked, failures = _exhaustive_check(stride)
    elapsed = time.time() - start
    print(f"States enumerated:  {checked:,} of {TOTAL_STATES:,} in {elapsed:.1f}s" + ("" if stride == 1 else " (sample only)"))
    print("Checked per state:  Tier 6 string <-> AST, sentence <-> AST, AST <-> Layer 8 / 11 / 12 code")
    print("Checked overall:    every checked state has a distinct Tier 6 string, sentence and Layer 8 / 11 / 12 code")
    for f in failures:
        print(f"  FAIL: {f}")
    assert not failures, "exhaustive bijection check failed"
    if stride == 1:
        print("Result:             PASS (bijection AST <-> code holds on all 294,912 states of this grammar)")
    else:
        print(f"Result:             PASS on the {checked:,} sampled states; this is a sample, run `symlang proof` (no --quick) for all of them")

    print("\n[STEP 3: OUT-OF-DOMAIN INPUT IS REJECTED, NOT REMAPPED]")
    bad_ast = Tier6Parser.parse(SAMPLE_TIER6.replace("▶M⇒", "▶ZZ⇒"))  # parses, but 'ZZ' is not a trigger_entity value
    rejections = [
        ("state id 294,912 (one past the end) -> Layer 11/12", _rejects(MetaTierCodec._state_id_to_ast, TOTAL_STATES)),
        ("code point U+0041 (below the Layer 12 range)", _rejects(MetaTierCodec.decode_layer12_singular, "A")),
        ("slot value 'ZZ' outside its domain -> Layer 11", _rejects(MetaTierCodec.encode_layer11_triad, bad_ast)),
    ]
    for desc, rejected in rejections:
        print(f"  {'rejected' if rejected else 'NOT REJECTED'}: {desc}")
    assert all(r for _, r in rejections)

    print("\n[STEP 4: THE ONE SAMPLE SENTENCE IN EACH LAYER (characters AND tokens)]")
    counter_name, count = get_token_counter("cl100k_base")
    ast = Tier6Parser.parse(SAMPLE_TIER6)
    canonical = expand_frame(ast)
    layers = [
        ("Canonical English sentence", canonical, None),
        ("Tier 6 SymOperator", SAMPLE_TIER6, Tier6Parser.parse),
        ("Layer 7:  SymTensor", MetaTierCodec.encode_layer7_tensor(ast), MetaTierCodec.decode_layer7_tensor),
        ("Layer 8:  SymRadix", MetaTierCodec.encode_layer8_radix(ast), MetaTierCodec.decode_layer8_radix),
        ("Layer 9A: SymBase85", MetaTierCodec.encode_layer9a_base85(ast), MetaTierCodec.decode_layer9a_base85),
        ("Layer 9B: SymRune", MetaTierCodec.encode_layer9b_runes(ast), MetaTierCodec.decode_layer9b_runes),
        ("Layer 10: SymGodel", str(MetaTierCodec.encode_layer10_godel(ast)), lambda s: MetaTierCodec.decode_layer10_godel(int(s))),
        ("Layer 11: SymTriad", MetaTierCodec.encode_layer11_triad(ast), MetaTierCodec.decode_layer11_triad),
        ("Layer 12: SymSingular", MetaTierCodec.encode_layer12_singular(ast), MetaTierCodec.decode_layer12_singular),
    ]
    raw_tokens = count(canonical)
    print(f"Token counts: {counter_name}. Compare tokens to the raw sentence; characters are not tokens.")
    print("-" * 105)
    print(f"{'Layer / representation':<28} | {'Chars':>5} | {'UTF-8 B':>7} | {'Tokens':>6} | {'vs raw tokens':>13} | {'Round trip':<10} | Output")
    print("-" * 105)
    for name, encoded, decoder_fn in layers:
        if decoder_fn is None:
            round_trip = "n/a"
        else:
            ok = decoder_fn(encoded).emit_tier6() == SAMPLE_TIER6
            assert ok, f"round trip failed on {name}"
            round_trip = "exact"
        tokens = count(encoded)
        change = f"{(tokens / raw_tokens - 1.0) * 100.0:+.1f}%"
        shown = encoded if len(encoded) <= 40 else encoded[:37] + "..."
        print(f"{name:<28} | {len(encoded):>5} | {len(encoded.encode('utf-8')):>7} | {tokens:>6} | {change:>13} | {round_trip:<10} | {shown}")
    print("-" * 105)
    print("The sample is state 0 (the first value of every slot), which is why Layer 11 is '!!!' and Layer 12 is U+10000.")

    print("\n" + "=" * 105)
    print(SCOPE_DISCLAIMER)
    print("=" * 105 + "\n")
    return True


if __name__ == "__main__":
    import sys

    run_formal_proof(stride=61 if "--quick" in sys.argv[1:] else 1)
