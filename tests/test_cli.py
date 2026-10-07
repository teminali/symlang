"""Tests for the command line, proof.py, run_pipeline_demo.py and index.html.

What they pin down: Layers 11 and 12 are reachable from `meta-decode`, out-of-domain input is an error (exit code 2),
`encode` reports the router's raw-vs-chosen tokens without ever showing a negative saving, and the user-facing text says
only what is measured (the frame result holds for ONE 294,912-sentence grammar; generic codecs cost tokens).
"""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cli  # noqa: E402
from symlang.lossless import LosslessSymbolicCodec  # noqa: E402
import proof  # noqa: E402
from symlang.decoder import expand_frame  # noqa: E402
from symlang.meta_tier import TOTAL_STATES, MetaTierCodec  # noqa: E402
from symlang.pipeline import NeuroSymbolicPipeline  # noqa: E402

CANONICAL = (
    "While the model starts, your PC can be slow or stop responding for 1-3 minutes "
    "(longest the first time). Strata loads 35-55 GB into your RAM and locks part of "
    "it for the graphics card. This is normal. Wait, and don't close the window. "
    "The window shows what Strata is doing."
)
SAMPLE_TIER6 = "▶M⇒PC~0 1-3m(1st:max).S:35-55G→RAM+GPU.✓ok.⏳¬✕win:win=👁S"
STATE_IDS = [0, 1, 84, 85, 7225, 12345, 147456, TOTAL_STATES - 1]

# Phrases the owner asked to remove or scope. Scanned in the user-facing files, case-insensitively.
STALE_PHRASES = [
    "99.63",
    "100% proven",
    "100% mathematically",
    "proven bijective",
    "zero information loss",
    "zero loss",
    "2.3x",
    "production-ready",
    "production ready",
    "theoretical limit reached",
]


def run_main(*argv):
    """Call cli.main(argv) in-process; returns (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def ast_for(state_id):
    return MetaTierCodec._state_id_to_ast(state_id)


class TestMetaDecodeLayers(unittest.TestCase):
    def test_triad_round_trip(self):
        for state_id in STATE_IDS:
            with self.subTest(state_id=state_id):
                ast = ast_for(state_id)
                triad = MetaTierCodec.encode_layer11_triad(ast)
                code, out, err = run_main("meta-decode", triad, "--layer", "triad")
                self.assertEqual(code, 0, err)
                self.assertIn(ast.emit_tier6(), out)
                self.assertIn(f"state {state_id:,} of {TOTAL_STATES:,}", out)

    def test_singular_round_trip(self):
        for state_id in STATE_IDS:
            with self.subTest(state_id=state_id):
                ast = ast_for(state_id)
                glyph = MetaTierCodec.encode_layer12_singular(ast)
                code, out, err = run_main("meta-decode", glyph, "--layer", "singular")
                self.assertEqual(code, 0, err)
                self.assertIn(ast.emit_tier6(), out)
                self.assertIn(f"state {state_id:,} of {TOTAL_STATES:,}", out)

    def test_existing_layers_still_round_trip(self):
        ast = ast_for(12345)
        cases = {
            "tensor": MetaTierCodec.encode_layer7_tensor(ast),
            "radix": MetaTierCodec.encode_layer8_radix(ast),
            "base85": MetaTierCodec.encode_layer9a_base85(ast),
            "rune": MetaTierCodec.encode_layer9b_runes(ast),
            "godel": str(MetaTierCodec.encode_layer10_godel(ast)),
        }
        for layer, text in cases.items():
            with self.subTest(layer=layer):
                code, out, err = run_main("meta-decode", text, "--layer", layer)
                self.assertEqual(code, 0, err)
                self.assertIn(ast.emit_tier6(), out)

    def test_output_makes_no_unmeasured_claim(self):
        _, out, _ = run_main("meta-decode", "!!!", "--layer", "triad")
        self.assertNotIn("100%", out)
        self.assertNotIn("Proven", out)


class TestMetaDecodeOutOfDomain(unittest.TestCase):
    def assert_rejected(self, text, layer, fragment=None):
        code, out, err = run_main("meta-decode", text, "--layer", layer)
        self.assertEqual(code, cli.EXIT_BAD_INPUT, (text, layer, out, err))
        self.assertEqual(cli.EXIT_BAD_INPUT, 2)
        self.assertTrue(err.startswith("error:"), err)
        self.assertNotIn("Reconstructed", out)
        if fragment:
            self.assertIn(fragment, err)

    def test_triad_wrong_length(self):
        self.assert_rejected("!!", "triad", "exactly 3")
        self.assert_rejected("!!!!", "triad", "exactly 3")

    def test_triad_state_id_past_the_end(self):
        # 85^3 = 614,125 triads exist but only 294,912 are states: the next id after the last state must be refused.
        val, chars = TOTAL_STATES, []
        for _ in range(3):
            chars.append(chr(33 + val % 85))
            val //= 85
        self.assert_rejected("".join(reversed(chars)), "triad", "outside")

    def test_triad_non_canonical_alias_is_refused(self):
        # '!v!' = digits (0, 85, 0): the raw decoder maps it onto state 7225 (same as '"!!'). The CLI accepts only the canonical form.
        self.assert_rejected("!v!", "triad", "not a canonical triad")

    def test_singular_wrong_length_and_range(self):
        self.assert_rejected("ab", "singular", "exactly 1")
        self.assert_rejected("", "singular", "exactly 1")
        self.assert_rejected("A", "singular", "outside")  # U+0041 is below U+10000
        self.assert_rejected(chr(0x10000 + TOTAL_STATES), "singular", "outside")  # one past the last state

    def test_godel_not_an_integer(self):
        self.assert_rejected("abc", "godel", "integer")

    def test_malformed_base85_is_an_error_not_a_traceback(self):
        self.assert_rejected("§\u0000\u0001", "base85")

    def test_exit_code_survives_a_real_process(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "cli.py"), "meta-decode", "A", "--layer", "singular"],
            capture_output=True, text=True, cwd=ROOT,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("error:", proc.stderr)


class TestMetaEncode(unittest.TestCase):
    def test_reports_triad_and_singular(self):
        code, out, err = run_main("meta-encode", SAMPLE_TIER6)
        self.assertEqual(code, 0, err)
        self.assertIn("Layer 11 (SymTriad):   !!!", out)
        self.assertIn("U+10000", out)
        self.assertIn("294,912", out)  # the scope note

    def test_text_outside_the_grammar_is_an_error(self):
        code, out, err = run_main("meta-encode", "it is fine; you can close the window")
        self.assertEqual(code, 2)
        self.assertIn("error:", err)

    def test_tier6_with_unknown_slot_value_gets_no_layer_11_12(self):
        code, out, err = run_main("meta-encode", SAMPLE_TIER6.replace("▶M⇒", "▶ZZ⇒"))
        self.assertEqual(code, 0, err)
        self.assertIn("Layer 11/12:           not available", out)


class TestEncodeRouterReport(unittest.TestCase):
    def payload_saving(self, out):
        m = re.search(r"Payload:\s+(\d+) tokens \((-?\d+\.\d)% saved", out)
        self.assertIsNotNone(m, out)
        return int(m.group(1)), float(m.group(2))

    def test_arbitrary_text_is_direct_with_zero_saving(self):
        for text in (
            "it is fine; you can close the window",
            "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)",
            '{"status": "ok", "items": [1, 2, 3]}',
        ):
            with self.subTest(text=text):
                code, out, err = run_main("encode", text, "--tier", "router")
                self.assertEqual(code, 0, err)
                self.assertIn("Direct (no saving found)", out)
                tokens, saving = self.payload_saving(out)
                self.assertEqual(saving, 0.0)
                raw = NeuroSymbolicPipeline().count_tokens(text)
                self.assertEqual(tokens, raw)

    def test_default_encode_also_reports_router_first(self):
        code, out, err = run_main("encode", "it is fine; you can close the window")
        self.assertEqual(code, 0, err)
        self.assertIn("--- Router", out)
        self.assertLess(out.index("--- Router"), out.index("--- Tier 1"))
        self.assertIn("no decoder guarantee", out)

    def test_candidates_that_cost_more_are_shown_with_a_plus_sign(self):
        _, out, _ = run_main("encode", "it is fine; you can close the window", "--tier", "router")
        for key in ("base85", "runes"):
            line = next(l for l in out.splitlines() if l.strip().startswith(key))
            self.assertRegex(line, r"\(\s*\+\d+\.\d%\)")
            self.assertIn("not eligible", line)

    def test_canonical_sentence_takes_the_frame_and_numbers_match_the_pipeline(self):
        code, out, err = run_main("encode", CANONICAL, "--tier", "router")
        self.assertEqual(code, 0, err)
        r = NeuroSymbolicPipeline().prepare_llm_input(CANONICAL)
        self.assertEqual(r["mode"], "frame_radix")
        self.assertIn(r["mode_used"], out)
        tokens, saving = self.payload_saving(out)
        self.assertEqual(tokens, r["compressed_tokens"])
        self.assertAlmostEqual(saving, r["input_token_saving_pct"], places=1)
        self.assertGreaterEqual(saving, 0.0)
        # The one-shot whole prompt costs MORE than raw; it must be reported as a plus, not hidden or called a saving.
        self.assertRegex(out, r"Whole prompt:\s+\d+ tokens incl\. protocol line \(\+\d+\.\d% tokens vs raw")

    def test_near_miss_of_the_canonical_sentence_goes_direct(self):
        code, out, _ = run_main("encode", CANONICAL.replace("don't close", "you can close"), "--tier", "router")
        self.assertEqual(code, 0)
        self.assertIn("Direct (no saving found)", out)

    def test_never_a_negative_payload_saving(self):
        for text in ("a", "hello world", CANONICAL, "x" * 500):
            with self.subTest(text=text[:20]):
                _, out, _ = run_main("encode", text, "--tier", "router")
                _, saving = self.payload_saving(out)
                self.assertGreaterEqual(saving, 0.0)

    def test_lossless_tier_reports_whether_this_input_round_trips(self):
        _, out, _ = run_main("encode", "plain ascii text", "--tier", "lossless")
        self.assertIn("Round trip for THIS input: exact", out)
        # non-ASCII used to be corrupted (and reported as "NOT exact"); it now round-trips, at 4 bytes per character
        _, out, _ = run_main("encode", "café 中文 🚀", "--tier", "lossless")
        self.assertIn("Round trip for THIS input: exact", out)
        self.assertNotIn("NOT exact", out)
        glyphs, _ = LosslessSymbolicCodec.encode_codebook_tokens("café 中文 🚀")
        code, out, _ = run_main("decode", glyphs, "--mode", "lossless")
        self.assertEqual(code, 0)
        self.assertIn("café 中文 🚀", out)

    def test_empty_text_is_an_error(self):
        code, _, err = run_main("encode")
        self.assertEqual(code, 2)
        self.assertIn("error:", err)


class TestProofCommand(unittest.TestCase):
    def test_quick_proof_states_its_scope(self):
        code, out, err = run_main("proof", "--quick")
        self.assertEqual(code, 0, err)
        self.assertIn("WHAT THIS DOES NOT PROVE", out)
        self.assertIn("ONE pre-agreed grammar of 294,912 sentences", out)
        self.assertIn("arbitrary text", out)
        self.assertIn("token cost", out)
        self.assertIn("MORE LLM tokens than the raw text", out)
        self.assertIn("1.8-3.3x", out)
        self.assertIn("NOT exhaustive", out)  # the sample run must label itself
        self.assertIn("18.1699 bits", out)  # the true entropy of the grammar is kept
        for phrase in STALE_PHRASES:
            self.assertNotIn(phrase, out.lower())

    def test_proof_script_prints_sample_sentence_tokens_not_just_characters(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertTrue(proof.run_formal_proof(stride=997))
        out = buf.getvalue()
        self.assertRegex(out, r"Layer 12: SymSingular\s+\|\s+1 \|\s+4 \|\s+4 \|")  # 1 char, 4 bytes, 4 tokens
        self.assertRegex(out, r"Layer 11: SymTriad\s+\|\s+3 \|\s+3 \|\s+1 \|")

    @unittest.skipUnless(os.environ.get("SYMLANG_EXHAUSTIVE") == "1", "set SYMLANG_EXHAUSTIVE=1 to enumerate all 294,912 states (10-30 s)")
    def test_exhaustive_bijection_over_every_state(self):
        checked, failures = proof._exhaustive_check(1)
        self.assertEqual(checked, TOTAL_STATES)
        self.assertEqual(failures, [])

    def test_the_check_can_fail(self):
        # Corrupt expand_frame: the check has to report it rather than pass vacuously.
        original = proof.expand_frame
        proof.expand_frame = lambda ast: "not a frame sentence"
        try:
            _, failures = proof._exhaustive_check(10007)
        finally:
            proof.expand_frame = original
        self.assertTrue(failures)
        _, failures = proof._exhaustive_check(10007)
        self.assertEqual(failures, [])


class TestUserFacingText(unittest.TestCase):
    def read(self, name):
        with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
            return fh.read()

    def test_no_stale_claims_in_owned_files(self):
        for name in ("cli.py", "proof.py", "index.html", "run_pipeline_demo.py"):
            text = self.read(name).lower()
            for phrase in STALE_PHRASES:
                with self.subTest(file=name, phrase=phrase):
                    self.assertNotIn(phrase, text)

    def test_demo_script_labels_the_llm_as_simulated(self):
        import run_pipeline_demo

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run_pipeline_demo.main()
        text = out.getvalue()
        self.assertIn("SIMULATED", text)
        self.assertIn("hardcoded", text)
        self.assertIn("costs MORE than raw", text)  # one-shot, protocol line counted
        self.assertIn("Direct (no saving found)", text)
        self.assertNotIn("Saved", text)

    def test_index_html_scope_text(self):
        html = self.read("index.html")
        for needle in ("294,912", "Decoder guarantee", "1.8-3.3x", "0.9-1.6x", "Direct (0% saving)", "no network"):
            with self.subTest(needle=needle):
                self.assertIn(needle, html)
        self.assertNotRegex(html, r"fetch\(|XMLHttpRequest|WebSocket")


class TestIndexHtmlTable(unittest.TestCase):
    """The page's static table must equal what the Python code emits and what tiktoken counts (drift guard)."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "index.html"), encoding="utf-8") as fh:
            html = fh.read()
        block = html[html.index("const TIERS = ["): html.index("// Characters = Unicode code points")]
        pattern = re.compile(
            r'id: "(?P<id>\w+)",\s*name: "[^"]*",\s*text: (?P<text>"(?:[^"\\]|\\.)*"),\s*'
            r'philosophy: "[^"]*",\s*cl100k: (?P<c>\d+),\s*o200k: (?P<o>\d+),'
        )
        cls.rows = {
            m["id"]: {"text": json.loads(m["text"]), "cl100k": int(m["c"]), "o200k": int(m["o"])}
            for m in pattern.finditer(block)
        }

    def expected_texts(self):
        from symlang.encoder import SymbolicEncoder
        from symlang.meta_tier import Tier6Parser

        ast = Tier6Parser.parse(SAMPLE_TIER6)
        tiers = SymbolicEncoder().encode_all_tiers(CANONICAL)
        return {
            "orig": CANONICAL,
            "tier1": tiers["tier1_telegraph"],
            "tier2": tiers["tier2_symlogic_ascii"],
            "tier3": tiers["tier3_symnano"],
            "tier6": SAMPLE_TIER6,
            "tier7": MetaTierCodec.encode_layer7_tensor(ast),
            "tier8": MetaTierCodec.encode_layer8_radix(ast),
            "tier9a": MetaTierCodec.encode_layer9a_base85(ast),
            "tier9b": MetaTierCodec.encode_layer9b_runes(ast),
            "tier10": str(MetaTierCodec.encode_layer10_godel(ast)),
            "tier11": MetaTierCodec.encode_layer11_triad(ast),
            "tier12": MetaTierCodec.encode_layer12_singular(ast),
        }

    def test_all_rows_parsed(self):
        self.assertEqual(set(self.rows), set(self.expected_texts()))

    def test_texts_match_the_code(self):
        for key, expected in self.expected_texts().items():
            with self.subTest(row=key):
                self.assertEqual(self.rows[key]["text"], expected)

    def test_demo_sentence_is_state_zero(self):
        self.assertEqual(expand_frame(ast_for(0)), CANONICAL)

    def test_token_counts_match_tiktoken(self):
        try:
            import tiktoken
            cl, o2 = tiktoken.get_encoding("cl100k_base"), tiktoken.get_encoding("o200k_base")
        except Exception:
            self.skipTest("tiktoken encodings unavailable")
        for key, row in self.rows.items():
            with self.subTest(row=key):
                self.assertEqual(row["cl100k"], len(cl.encode(row["text"], disallowed_special=())))
                self.assertEqual(row["o200k"], len(o2.encode(row["text"], disallowed_special=())))


if __name__ == "__main__":
    unittest.main()
