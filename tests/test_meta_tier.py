import unittest
from symlang.meta_tier import (
    Tier6Parser,
    MetaTierCodec,
    verify_mathematical_roundtrip,
)

class TestMetaTierCodec(unittest.TestCase):
    def setUp(self):
        self.tier6_sample = "▶M⇒PC~0 1-3m(1st:max).S:35-55G→RAM+GPU.✓ok.⏳¬✕win:win=👁S"

    def test_parser_and_emitter(self):
        ast = Tier6Parser.parse(self.tier6_sample)
        self.assertEqual(ast.emit_tier6(), self.tier6_sample)

    def test_layer7_tensor_roundtrip(self):
        ast = Tier6Parser.parse(self.tier6_sample)
        encoded = MetaTierCodec.encode_layer7_tensor(ast)
        decoded_ast = MetaTierCodec.decode_layer7_tensor(encoded)
        self.assertEqual(decoded_ast.emit_tier6(), self.tier6_sample)

    def test_layer8_radix_roundtrip(self):
        ast = Tier6Parser.parse(self.tier6_sample)
        encoded = MetaTierCodec.encode_layer8_radix(ast)
        decoded_ast = MetaTierCodec.decode_layer8_radix(encoded)
        self.assertEqual(decoded_ast.emit_tier6(), self.tier6_sample)

    def test_layer11_triad_roundtrip(self):
        ast = Tier6Parser.parse(self.tier6_sample)
        encoded = MetaTierCodec.encode_layer11_triad(ast)
        self.assertEqual(len(encoded), 3)
        decoded_ast = MetaTierCodec.decode_layer11_triad(encoded)
        self.assertEqual(decoded_ast.emit_tier6(), self.tier6_sample)

    def test_layer12_singular_roundtrip(self):
        ast = Tier6Parser.parse(self.tier6_sample)
        encoded = MetaTierCodec.encode_layer12_singular(ast)
        self.assertEqual(len(encoded), 1)
        decoded_ast = MetaTierCodec.decode_layer12_singular(encoded)
        self.assertEqual(decoded_ast.emit_tier6(), self.tier6_sample)

    def test_verify_roundtrip_function(self):
        res = verify_mathematical_roundtrip(self.tier6_sample)
        self.assertTrue(res["all_layers_mathematically_proven"])

    def test_out_of_domain_and_out_of_range_raise_instead_of_aliasing(self):
        from symlang.meta_tier import TOTAL_STATES, Tier6AST

        self.assertEqual(TOTAL_STATES, 294912)
        ast = Tier6Parser.parse(self.tier6_sample)
        bad = Tier6AST.from_vector(["ZZZ"] + ast.to_vector()[1:])  # slot value outside the closed domain
        with self.assertRaises(ValueError):
            MetaTierCodec.encode_layer12_singular(bad)
        with self.assertRaises(ValueError):
            MetaTierCodec.encode_layer11_triad(bad)
        # ids >= TOTAL_STATES used to silently alias onto valid states
        with self.assertRaises(ValueError):
            MetaTierCodec.decode_layer12_singular(chr(0x10000 + TOTAL_STATES))
        with self.assertRaises(ValueError):
            MetaTierCodec.decode_layer11_triad("uuu")

    def test_triad_decoder_is_strict_one_state_one_code(self):
        # '!v!' = digits (0, 85, 0) used to alias onto state 7225 (same as '"!!'); a decoder that accepts several
        # spellings of one state breaks the bijection, so every non-canonical spelling must raise.
        canonical = MetaTierCodec.encode_layer11_triad(MetaTierCodec._state_id_to_ast(7225))
        self.assertEqual(canonical, '"!!')
        self.assertEqual(MetaTierCodec.decode_layer11_triad(canonical), MetaTierCodec._state_id_to_ast(7225))
        for alias in ["!v!", "!!v", "v!!", "!~!", "! !", "!\u00e9!", "!\U0001F680!"]:
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(ValueError, "not a canonical triad"):
                    MetaTierCodec.decode_layer11_triad(alias)
        for bad_len in ["", "!", "!!", "!!!!"]:
            with self.assertRaises(ValueError):
                MetaTierCodec.decode_layer11_triad(bad_len)
        with self.assertRaisesRegex(ValueError, "outside"):
            MetaTierCodec.decode_layer11_triad("uuu")  # canonical characters but past the last state

    def test_triad_is_a_bijection_between_accepted_codes_and_states(self):
        # exhaustive over all 85^3 spellings: exactly TOTAL_STATES are accepted and each maps to a distinct state
        from symlang.meta_tier import TOTAL_STATES

        accepted = {}
        digits = [chr(33 + d) for d in range(85)]
        for a in digits:
            for b in digits:
                for c in digits:
                    code = a + b + c
                    try:
                        ast = MetaTierCodec.decode_layer11_triad(code)
                    except ValueError:
                        continue
                    accepted[code] = MetaTierCodec._ast_to_state_id(ast)
        self.assertEqual(len(accepted), TOTAL_STATES)
        self.assertEqual(len(set(accepted.values())), TOTAL_STATES)
        for code, sid in list(accepted.items())[::997]:
            self.assertEqual(MetaTierCodec.encode_layer11_triad(MetaTierCodec._state_id_to_ast(sid)), code)

    def test_state_id_bijection_over_a_sample_of_the_grammar(self):
        from symlang.meta_tier import TOTAL_STATES

        for sid in range(0, TOTAL_STATES, 97):
            ast = MetaTierCodec._state_id_to_ast(sid)
            self.assertEqual(MetaTierCodec._ast_to_state_id(ast), sid)
            self.assertEqual(MetaTierCodec.decode_layer12_singular(MetaTierCodec.encode_layer12_singular(ast)), ast)
            self.assertEqual(MetaTierCodec.decode_layer11_triad(MetaTierCodec.encode_layer11_triad(ast)), ast)


if __name__ == "__main__":
    unittest.main()
