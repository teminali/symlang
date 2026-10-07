import unittest

from symlang.pipeline import NeuroSymbolicPipeline, verify_roundtrip

CANONICAL = (
    "While the model starts, your PC can be slow or stop responding for 1-3 minutes "
    "(longest the first time). Strata loads 35-55 GB into your RAM and locks part of "
    "it for the graphics card. This is normal. Wait, and don't close the window. "
    "The window shows what Strata is doing."
)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = NeuroSymbolicPipeline()
        self.sample_text = CANONICAL

    def test_pipeline_turn_execution(self):
        res = self.pipeline.execute_turn(self.sample_text)
        self.assertIn("input_stage", res)
        self.assertIn("output_stage", res)
        self.assertIn("total_token_metrics", res)

        # No llm_caller: the LLM is an explicit identity echo, and the result says so.
        self.assertTrue(res["simulated_llm"])
        self.assertEqual(res["llm_intermediate_reply"], res["input_stage"]["compressed_symbol"])

        # The canonical sentence takes the frame (measured, not asserted from a constant) and round trips.
        in_stage = res["input_stage"]
        self.assertEqual(in_stage["mode"], "frame_radix")
        self.assertTrue(in_stage["roundtrip_verified"])
        self.assertTrue(in_stage["roundtrip_exact"])
        self.assertEqual(res["output_stage"]["human_readable_output"], self.sample_text)

        # The saving is whatever the tokenizer measures: it must equal the recomputed value and be non-negative.
        count = self.pipeline.count_tokens
        expected_in = 100.0 * (1 - count(in_stage["compressed_symbol"]) / count(self.sample_text))
        self.assertAlmostEqual(in_stage["input_token_saving_pct"], expected_in, places=6)
        self.assertGreaterEqual(res["total_token_metrics"]["overall_token_saving_pct"], 0.0)

    def test_scaffold_is_reported_honestly(self):
        res = self.pipeline.execute_turn(self.sample_text)
        tm = res["total_token_metrics"]
        # The protocol line of the prompt costs tokens; the figure that includes it must be reported and be lower.
        self.assertIn("overall_saving_pct_with_prompt_scaffold", tm)
        self.assertLess(tm["overall_saving_pct_with_prompt_scaffold"], tm["overall_token_saving_pct"])

    def test_real_llm_caller_path_is_unchanged(self):
        seen = {}

        def caller(prompt):
            seen["prompt"] = prompt
            return "plain text answer"

        res = self.pipeline.execute_turn("hello there", llm_caller=caller)
        self.assertFalse(res["simulated_llm"])
        self.assertEqual(seen["prompt"], res["input_stage"]["llm_prompt"])
        self.assertEqual(res["output_stage"]["mode_used"], "Direct")
        self.assertEqual(res["output_stage"]["human_readable_output"], "plain text answer")

    def test_verify_roundtrip_helper(self):
        self.assertTrue(verify_roundtrip("a  b", "x", lambda s: "a b"))          # whitespace-normalised
        self.assertFalse(verify_roundtrip("a  b", "x", lambda s: "a b", normalize=None))  # exact
        self.assertFalse(verify_roundtrip("a", "x", lambda s: "b"))
        self.assertFalse(verify_roundtrip("a", "x", lambda s: 1 / 0))             # decode error is a failure


if __name__ == "__main__":
    unittest.main()
