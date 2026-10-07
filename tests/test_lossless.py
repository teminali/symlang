"""LosslessSymbolicCodec must round-trip EVERY str exactly (it used to corrupt non-ASCII text)."""

import random
import unittest

from symlang.lossless import SYSTEM_CODEBOOK, LosslessSymbolicCodec

C = LosslessSymbolicCodec

CODEBOOK_SENTENCE = (
    "While the model starts, your PC can be slow or stop responding for 1-3 minutes "
    "(longest the first time). Strata loads 35-55 GB into your RAM and locks part of "
    "it for the graphics card. This is normal. Wait, and don't close the window. "
    "The window shows what Strata is doing."
)


def roundtrip(text: str) -> str:
    glyphs, raw = C.encode_codebook_tokens(text)
    # all three decode paths must agree
    assert C.decode_codebook_tokens(raw) == C.decode_unicode_glyphs(glyphs)
    return C.decode_unicode_glyphs(glyphs)


class TestLosslessNonAscii(unittest.TestCase):
    def test_the_reported_case(self):
        text = "héllo wörld, 日本語, emoji 🚀, tabs\t\tand \r\n"
        self.assertEqual(roundtrip(text), text)
        self.assertTrue(C.roundtrip_test(text))

    def test_each_kind_of_non_ascii(self):
        for text in [
            "é",  # Latin-1
            "日本語",  # BMP, 3-byte UTF-8
            "🚀",  # astral plane
            "é",  # combining acute accent
            "👨‍👩‍👧",  # ZWJ emoji sequence
            "\x80\xffĀ߿ࠀ￿\U00010000\U0010ffff",  # boundaries
            "\ud800",  # a lone surrogate is still a str: must not be lost either
        ]:
            with self.subTest(text=text):
                self.assertEqual(roundtrip(text), text)

    def test_ascii_controls_and_whitespace(self):
        text = "".join(chr(i) for i in range(128)) + "\r\n\r\n\t "
        self.assertEqual(roundtrip(text), text)

    def test_empty(self):
        self.assertEqual(roundtrip(""), "")

    def test_codebook_phrases_still_compress_and_mix_with_non_ascii(self):
        glyphs, raw = C.encode_codebook_tokens(CODEBOOK_SENTENCE)
        self.assertLess(len(raw), len(CODEBOOK_SENTENCE) / 3)
        self.assertEqual(C.decode_unicode_glyphs(glyphs), CODEBOOK_SENTENCE)
        mixed = SYSTEM_CODEBOOK[0] + "日本 🚀 " + SYSTEM_CODEBOOK[5] + "é"
        self.assertEqual(roundtrip(mixed), mixed)

    def test_ascii_encoding_is_unchanged(self):
        # reference = the original (ASCII-only) packing: codebook index < 0x80, literal = 0x80 | ord(c)
        def legacy(text: str) -> bytes:
            out, i = bytearray(), 0
            while i < len(text):
                for idx, phrase in enumerate(SYSTEM_CODEBOOK):
                    if text[i:].startswith(phrase):
                        out.append(idx)
                        i += len(phrase)
                        break
                else:
                    out.append(0x80 | (ord(text[i]) & 0x7F))
                    i += 1
            return bytes(out)

        rng = random.Random(5)
        samples = [CODEBOOK_SENTENCE, "Wait, and don't close the window. A\r\n", "This is normal. x", "plain ascii 123 ~!"]
        for _ in range(300):
            samples.append("".join(rng.choice([rng.choice(SYSTEM_CODEBOOK), chr(rng.randrange(128))]) for _ in range(12)))
        for text in samples:
            glyphs, raw = C.encode_codebook_tokens(text)
            self.assertEqual(raw, legacy(text), text)
            self.assertEqual(glyphs, "".join(chr(0x2400 + b) for b in raw))

    def test_malformed_input_raises_value_error(self):
        for bad in [bytes([0x7F]), bytes([0x7F, 0x00, 0x01]), bytes([len(SYSTEM_CODEBOOK)]), bytes([0x7E])]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    C.decode_codebook_tokens(bad)
        with self.assertRaises(ValueError):
            C.decode_unicode_glyphs("A")  # not a glyph in U+2400..U+24FF
        # an escaped code point past U+10FFFF is not a character
        with self.assertRaises(ValueError):
            C.decode_codebook_tokens(bytes([0x7F, 0x11, 0x00, 0x00]))


class TestLosslessProperty(unittest.TestCase):
    """Seeded property test: random strings drawn from many Unicode regions round-trip exactly."""

    ALPHABETS = [
        [chr(i) for i in range(0x20, 0x7F)] + ["\t", "\r", "\n", "\x00", "\x7f"],  # ASCII
        [chr(i) for i in range(0x80, 0x250)],  # Latin supplements and extended
        [chr(i) for i in range(0x300, 0x370)],  # combining marks
        [chr(i) for i in range(0x3040, 0x30A0)] + [chr(i) for i in range(0x4E00, 0x4E80)],  # kana and CJK
        [chr(i) for i in range(0x1F300, 0x1F5FF)],  # emoji (astral)
        ["‍", "️", "‮", "﻿", " "],  # joiners, variation selector, bidi, BOM, line sep
    ]

    @staticmethod
    def _random_text(rng: random.Random) -> str:
        parts = []
        for _ in range(rng.randint(0, 40)):
            kind = rng.random()
            if kind < 0.15:
                parts.append(rng.choice(SYSTEM_CODEBOOK))  # codebook phrases interleaved with everything else
            elif kind < 0.25:
                parts.append(chr(rng.choice([0x7F, 0x80, 0xFF, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF])))
            else:
                alphabet = rng.choice(TestLosslessProperty.ALPHABETS)
                parts.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 8))))
        return "".join(parts)

    def test_random_unicode_round_trips(self):
        rng = random.Random(20260507)
        for n in range(3000):
            text = self._random_text(rng)
            self.assertEqual(roundtrip(text), text, f"case {n}: {text!r}")

    def test_random_code_points_including_lone_surrogates(self):
        rng = random.Random(7)
        for n in range(500):
            text = "".join(chr(rng.randrange(0, 0x110000)) for _ in range(rng.randint(0, 30)))
            self.assertEqual(roundtrip(text), text, f"case {n}")

    def test_utf8_encodable_strings_survive_a_byte_level_trip(self):
        rng = random.Random(99)
        for _ in range(300):
            text = self._random_text(rng)
            glyphs, raw = C.encode_codebook_tokens(text)
            self.assertEqual(C.decode_codebook_tokens(bytes(raw)).encode("utf-8"), text.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
