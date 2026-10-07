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

if __name__ == "__main__":
    unittest.main()
