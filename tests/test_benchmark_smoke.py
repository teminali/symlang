"""benchmark.py must stay runnable and must not print unmeasured claims."""

import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_benchmark(*args: str) -> str:
    # run as a script: `import benchmark` would resolve to the benchmark/ package, not benchmark.py
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "benchmark.py"), *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


class TestBenchmarkSmoke(unittest.TestCase):
    def test_default_sample_prints_only_measured_claims(self):
        out = run_benchmark()
        for banned in ("100%", "99.6", "zero loss", "zero-loss", "Proven Bijective", "lossless 100"):
            self.assertNotIn(banned.lower(), out.lower(), banned)
        self.assertIn("cl100k_base", out)
        self.assertIn("294,912", out)  # the frame result is scoped to the one grammar
        self.assertIn("Layer 11 SymTriad", out)
        self.assertIn("identical = True", out)
        self.assertIn("-98.6%", out)  # the signed, measured Layer 11 token change for the canonical sentence (3 chars, 1 token)

    def test_text_outside_the_grammar_skips_the_frame_rows(self):
        out = run_benchmark("--text", "héllo wörld, 日本語 🚀")
        self.assertIn("rows skipped", out)
        self.assertNotIn("Layer 11 SymTriad", out)
        self.assertNotIn("100%", out)
        self.assertIn("Lossless codebook glyphs", out)
        self.assertNotIn("NOT exact", out.split("Lossless codebook glyphs")[1].splitlines()[0])


if __name__ == "__main__":
    unittest.main()
