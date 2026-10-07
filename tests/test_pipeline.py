import unittest
from symlang.pipeline import NeuroSymbolicPipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = NeuroSymbolicPipeline()
        self.sample_text = (
            "While the model starts, your PC can be slow or stop responding for 1-3 minutes "
            "(longest the first time). Strata loads 35-55 GB into your RAM and locks part of "
            "it for the graphics card. This is normal. Wait, and don't close the window. "
            "The window shows what Strata is doing."
        )

    def test_pipeline_turn_execution(self):
        res = self.pipeline.execute_turn(self.sample_text)
        self.assertIn("input_stage", res)
        self.assertIn("output_stage", res)
        self.assertIn("total_token_metrics", res)
        
        # Verify token reduction is greater than 50%
        overall_saving = res["total_token_metrics"]["overall_token_saving_pct"]
        self.assertGreater(overall_saving, 50.0)
        
        # Verify human-readable output is generated
        human_text = res["output_stage"]["human_readable_output"]
        self.assertIn("Strata", human_text)
        self.assertIn("RAM", human_text)

if __name__ == "__main__":
    unittest.main()
