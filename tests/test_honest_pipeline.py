"""Honesty tests: the pipeline may only claim what it measures and verifies."""
import random
import unittest

from symlang.decoder import (
    FRAME_PHRASES,
    SymbolicDecoder,
    expand_frame,
    normalize_ws,
    parse_canonical_sentence,
)
from symlang.meta_tier import DOMAIN_SLOTS, TOTAL_STATES, MetaTierCodec, Tier6AST, Tier6Parser
from symlang.pipeline import NeuroSymbolicPipeline, verify_roundtrip
from symlang.universal_codec import UniversalExactCodec

CANONICAL = (
    "While the model starts, your PC can be slow or stop responding for 1-3 minutes "
    "(longest the first time). Strata loads 35-55 GB into your RAM and locks part of "
    "it for the graphics card. This is normal. Wait, and don't close the window. "
    "The window shows what Strata is doing."
)
CANNED_PHRASES = ("Please wait and do not close the window", "may experience slowness")
TIER6 = "▶M⇒PC~0 1-3m(1st:max).S:35-55G→RAM+GPU.✓ok.⏳¬✕win:win=👁S"

DIVERSE_STRINGS = [
    "",
    "a",
    "hi",
    "ok",
    "yes",
    "👍",
    "Hello, world!",
    "it is fine; you can close the window",
    "Please wait and do not close the window",
    "The model starts and Strata loads RAM.",
    "ram strata model starts",
    "model starts",
    CANONICAL,
    CANONICAL + " Thank you.",
    CANONICAL.replace("normal", "abnormal"),
    CANONICAL.replace("don't close", "close"),
    "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    "for (let i = 0; i < n; i++) {\n\tconsole.log(i);\n}\n",
    "SELECT id, name FROM users WHERE created_at > '2024-01-01' ORDER BY name;",
    '{"name": "symlang", "version": "1.0.0", "tags": ["nlp", "compression"], "nested": {"a": 1, "b": [1, 2, 3]}}',
    "# Title\n\n- item one\n- item two\n\n```python\nprint('x')\n```\n",
    "The quick brown fox jumps over the lazy dog. " * 20,
    "ab" * 500,
    "x" * 3000,
    "Ünïcödé text: naïve café, 東京, привет, مرحبا, 你好世界, 😀🎉🔥",
    "日本語のテキストです。これは圧縮のテストです。",
    "line1\r\nline2\r\n\r\n   indented\r\n",
    "\t\ttabs\tand  double  spaces   ",
    "A short sentence about nothing in particular.",
    "https://github.com/teminali/symlang/issues/1?x=1&y=2#frag",
    "0123456789" * 30,
    "".join(chr(0x4E00 + i) for i in range(40)),
    "".join(chr(0x10000 + i * 997) for i in range(10)),
    "Ω[M·PC·~0·1-3m·1st:max·S·35-55G·RAM·GPU·ok·win]",
    TIER6,
]
assert len(DIVERSE_STRINGS) >= 30


class TestRouterNeverWorseThanRaw(unittest.TestCase):
    def setUp(self):
        self.pipeline = NeuroSymbolicPipeline()
        self.count = self.pipeline.count_tokens

    def test_router_property_over_diverse_inputs(self):
        for text in DIVERSE_STRINGS:
            for readable_only in (True, False):
                with self.subTest(text=text[:40], readable_only=readable_only):
                    r = self.pipeline.prepare_llm_input(text, llm_readable_only=readable_only)
                    raw_tokens = self.count(text)
                    self.assertEqual(r["original_tokens"], raw_tokens)
                    self.assertLessEqual(r["compressed_tokens"], raw_tokens)
                    self.assertEqual(r["compressed_tokens"], self.count(r["compressed_symbol"]))
                    self.assertGreaterEqual(r["input_token_saving_pct"], 0.0)
                    if r["mode"] == "direct":
                        self.assertEqual(r["compressed_symbol"], text)
                        self.assertEqual(r["input_token_saving_pct"], 0.0)
                        self.assertTrue(r["mode_used"].startswith("Direct"))
                    # Whatever was chosen must round trip.
                    self.assertTrue(r["roundtrip_verified"])

    def test_nothing_beats_raw_is_reported_as_direct(self):
        r = self.pipeline.prepare_llm_input("def f(x):\n    return x + 1\n")
        self.assertEqual(r["mode_used"], "Direct (no saving found)")
        self.assertEqual(r["input_token_saving_pct"], 0.0)

    def test_generic_codecs_cost_more_tokens_and_are_not_picked(self):
        text = "The quick brown fox jumps over the lazy dog. " * 3
        r = self.pipeline.prepare_llm_input(text, llm_readable_only=False)
        by_key = {c["key"]: c for c in r["candidates"]}
        self.assertGreater(by_key["runes"]["tokens"], by_key["direct"]["tokens"])
        self.assertGreater(by_key["base85"]["tokens"], by_key["direct"]["tokens"])
        self.assertEqual(r["mode"], "direct")

    def test_opaque_forms_need_explicit_opt_in(self):
        text = "ab" * 2000  # zlib genuinely wins on tokens here
        default = self.pipeline.prepare_llm_input(text)
        self.assertEqual(default["mode"], "direct")  # an LLM cannot inflate DEFLATE
        opted = self.pipeline.prepare_llm_input(text, llm_readable_only=False)
        self.assertIn(opted["mode"], ("base85", "runes"))
        self.assertLess(opted["compressed_tokens"], opted["original_tokens"])
        self.assertTrue(opted["roundtrip_exact"])
        self.assertIn("not LLM-readable", opted["mode_used"])

    def test_injectable_token_counter(self):
        def fake(s):  # a "tokenizer" that charges 5 per character: changes the decision, proving it is consulted
            return 5 * len(s)

        p = NeuroSymbolicPipeline(token_counter=fake)
        r = p.prepare_llm_input(CANONICAL, llm_readable_only=False)
        self.assertEqual(r["original_tokens"], 5 * len(CANONICAL))
        self.assertEqual(r["compressed_tokens"], 5 * len(r["compressed_symbol"]))
        self.assertEqual(r["mode"], "frame_singular")  # 1 glyph is the cheapest under char-based counting


class TestRoundTripGuard(unittest.TestCase):
    def setUp(self):
        self.pipeline = NeuroSymbolicPipeline()

    def test_close_the_window_reversal_is_not_the_canned_text(self):
        text = "it is fine; you can close the window"
        r = self.pipeline.prepare_llm_input(text)
        self.assertEqual(r["mode"], "direct")
        self.assertEqual(r["compressed_symbol"], text)
        res = self.pipeline.execute_turn(text)
        out = res["output_stage"]["human_readable_output"]
        self.assertEqual(out, text)
        for canned in CANNED_PHRASES:
            self.assertNotIn(canned, out)
        # The decoder alone must not turn it into the opposite instruction either.
        self.assertEqual(SymbolicDecoder.decode_deterministic(text), text)

    def test_keyword_triggers_are_gone(self):
        for text in ("model starts", "ram strata", "The model starts; Strata loads RAM.", "35-55 Strata S: ⚡"):
            r = self.pipeline.prepare_llm_input(text)
            self.assertEqual(r["mode"], "direct", text)
            self.assertEqual(SymbolicDecoder.decode_deterministic(text), text)
        # 'telemetry' no longer forces the frame on arbitrary text.
        r = self.pipeline.prepare_llm_input("model starts and RAM Strata", domain_mode="telemetry")
        self.assertEqual(r["mode"], "direct")
        self.assertIn("not the canonical frame sentence", r["note"])

    def test_near_miss_of_canonical_sentence_never_takes_the_frame(self):
        variants = [
            CANONICAL.replace("normal", "fine"),
            CANONICAL.replace("1-3 minutes", "7 minutes"),
            CANONICAL.replace("don't close", "close"),
            CANONICAL.replace("Strata", "Strat"),
            CANONICAL[:-1],
            CANONICAL.lower(),
        ]
        for v in variants:
            with self.subTest(v=v[-60:]):
                r = self.pipeline.prepare_llm_input(v)
                self.assertEqual(r["mode"], "direct")
                self.assertEqual(r["compressed_symbol"], v)
                self.assertIsNone(parse_canonical_sentence(v))

    def test_canonical_sentence_uses_the_frame_and_round_trips(self):
        r = self.pipeline.prepare_llm_input(CANONICAL)
        self.assertEqual(r["mode"], "frame_radix")
        self.assertEqual(r["compressed_symbol"], "Ω[M·PC·~0·1-3m·1st:max·S·35-55G·RAM·GPU·ok·win]")
        self.assertTrue(r["roundtrip_verified"] and r["roundtrip_exact"])
        self.assertLess(r["compressed_tokens"], r["original_tokens"])
        back = self.pipeline.parse_llm_output(r["compressed_symbol"])["human_readable_output"]
        self.assertEqual(back, CANONICAL)

    def test_frame_is_whitespace_tolerant_but_reports_non_exact(self):
        spaced = CANONICAL.replace(". ", ".\n\n")
        r = self.pipeline.prepare_llm_input(spaced)
        self.assertEqual(r["mode"], "frame_radix")
        self.assertTrue(r["roundtrip_verified"])
        self.assertFalse(r["roundtrip_exact"])  # honest: whitespace was normalised

    def test_codebook_codes_need_opt_in_and_round_trip(self):
        r = self.pipeline.prepare_llm_input(CANONICAL, llm_readable_only=False)
        self.assertEqual(r["mode"], "frame_triad")
        self.assertEqual(len(r["compressed_symbol"]), 3)
        self.assertEqual(r["compressed_tokens"], 1)
        # without codebook=True the string "!!!" is just text; with it, it is the canonical sentence
        self.assertEqual(self.pipeline.parse_llm_output("!!!")["mode_used"], "Direct")
        out = self.pipeline.parse_llm_output(r["compressed_symbol"], codebook=True)
        self.assertEqual(out["human_readable_output"], CANONICAL)
        res = self.pipeline.execute_turn(CANONICAL, llm_readable_only=False)
        self.assertEqual(res["output_stage"]["human_readable_output"], CANONICAL)

    def test_arbitrary_inputs_never_negative_and_always_round_trip(self):
        rng = random.Random(7)
        alphabet = "abcdefghij klmnop\n\t(){}[];:,.'\"0123456789éü日😀"
        inputs = list(DIVERSE_STRINGS) + [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 400))) for _ in range(40)
        ]
        for text in inputs:
            with self.subTest(text=text[:30]):
                res = self.pipeline.execute_turn(text, llm_readable_only=False)
                self.assertTrue(res["simulated_llm"])
                self.assertGreaterEqual(res["input_stage"]["input_token_saving_pct"], 0.0)
                self.assertGreaterEqual(res["total_token_metrics"]["overall_token_saving_pct"], 0.0 - 1e-9)
                got = res["output_stage"]["human_readable_output"]
                if res["input_stage"]["roundtrip_exact"]:
                    # parse_llm_output strips the reply: compare with outer whitespace removed for Direct echoes
                    self.assertEqual(got, text.strip() if res["input_stage"]["mode"] == "direct" else text)
                    self.assertEqual(got.strip(), text.strip())
                else:
                    self.assertEqual(normalize_ws(got), normalize_ws(text))


class TestRuneReplies(unittest.TestCase):
    def setUp(self):
        self.pipeline = NeuroSymbolicPipeline()

    def test_rune_reply_decodes_through_parse_llm_output(self):
        for text in ("def f(x):\n    return x + 1\n", "plain prose, nothing special.", CANONICAL, "日本語 😀 mix", "ab" * 300):
            runes = UniversalExactCodec.encode_to_runes(text)
            self.assertTrue(all(ord(c) >= 0x10000 for c in runes))  # plane 1, not the CJK block
            out = self.pipeline.parse_llm_output(runes)
            self.assertEqual(out["mode_used"], "UniversalRunes -> Lossless Text")
            self.assertEqual(out["human_readable_output"], text)

    def test_rune_replies_with_odd_and_even_stream_lengths(self):
        seen = set()
        for n in range(1, 60):
            text = "".join(chr(97 + (i * 7 + n) % 26) for i in range(n))
            runes = UniversalExactCodec.encode_to_runes(text)
            seen.add(len(UniversalExactCodec.encode_to_base85(text)) % 2)
            self.assertEqual(self.pipeline.parse_llm_output(runes)["human_readable_output"], text)

    def test_garbage_plane1_text_falls_back_to_direct(self):
        for reply in ("Sure! 😀", "😀", "𐀀𐀁𐀂", "Here you go 🎉🎉 enjoy", "\U0001F600" * 5):
            out = self.pipeline.parse_llm_output(reply)
            self.assertEqual(out["mode_used"], "Direct", reply)
            self.assertEqual(out["human_readable_output"], reply.strip())

    def test_bad_radix_falls_back_to_direct(self):
        for reply in ("Ω[bad]", "Ω[X·Y·Z·a·b·c·d·e·f·g·h]", "Ω[", "the symbol is Ω[ nothing"):
            out = self.pipeline.parse_llm_output(reply)
            self.assertEqual(out["mode_used"], "Direct", reply)
            self.assertEqual(out["human_readable_output"], reply)

    def test_radix_inside_prose_is_expanded_in_place(self):
        out = self.pipeline.parse_llm_output("Status: Ω[M·PC·~0·1-3m·1st:max·S·35-55G·RAM·GPU·ok·win] end.")
        self.assertEqual(out["human_readable_output"], "Status: " + CANONICAL + " end.")


class TestDecoderIsAnExpansion(unittest.TestCase):
    def test_decode_is_derived_from_the_ast(self):
        ast = Tier6Parser.parse(TIER6)
        self.assertEqual(SymbolicDecoder.decode_deterministic(TIER6), CANONICAL)
        self.assertEqual(expand_frame(ast), CANONICAL)
        # change one slot -> the text changes accordingly (a lookup could not do this)
        other = Tier6AST.from_vector(["SYS", "HOST", "~", "2-4m", "cold", "SYS", "8-16G", "VRAM", "CPU", "ready", "ui"])
        text = SymbolicDecoder.decode_deterministic(other.emit_tier6())
        self.assertEqual(text, expand_frame(other))
        self.assertIn("the system starts", text)
        self.assertIn("8-16 GB", text)
        self.assertNotEqual(text, CANONICAL)

    def test_all_symbol_forms_decode_the_same(self):
        ast = Tier6Parser.parse(TIER6)
        for sym in (TIER6, MetaTierCodec.encode_layer8_radix(ast), MetaTierCodec.encode_layer7_tensor(ast)):
            self.assertEqual(SymbolicDecoder.decode_deterministic(sym), CANONICAL)

    def test_unknown_or_out_of_domain_input_is_returned_unchanged(self):
        out_of_domain = "Ω[Z·PC·~0·1-3m·1st:max·S·35-55G·RAM·GPU·ok·win]"
        for text in ("", "hello", "35-55 Strata S: ⚡", out_of_domain, TIER6.replace("35-55G", "99G")):
            self.assertEqual(SymbolicDecoder.decode_deterministic(text), text)

    def test_phrase_tables_are_invertible_and_cover_the_domain(self):
        for slot, values in DOMAIN_SLOTS:
            self.assertEqual(list(FRAME_PHRASES[slot]), list(values))
            phrases = list(FRAME_PHRASES[slot].values())
            self.assertEqual(len(set(phrases)), len(phrases), slot)

    def test_text_ast_text_is_identity_on_a_large_sample_of_the_grammar(self):
        for sid in range(0, TOTAL_STATES, 11):  # ~26.8k states; the exhaustive 294,912 run takes ~5s (run by hand)
            ast = MetaTierCodec._state_id_to_ast(sid)
            sentence = expand_frame(ast)
            self.assertEqual(parse_canonical_sentence(sentence), ast)
            self.assertTrue(verify_roundtrip(sentence, MetaTierCodec.encode_layer12_singular(ast),
                                             lambda s: expand_frame(MetaTierCodec.decode_layer12_singular(s))))


if __name__ == "__main__":
    unittest.main()
