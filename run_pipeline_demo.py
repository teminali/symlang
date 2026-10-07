#!/usr/bin/env python3
"""Live Execution Demo of the Bidirectional Neuro-Symbolic Pipeline.

Demonstrates:
[Human Input Text] -> Parser -> [Token-Compressed Symbol] -> LLM Context
LLM Generates -> [Token-Compressed Symbol] -> Parser -> [Human-Readable Text]
"""

import sys
from symlang.pipeline import NeuroSymbolicPipeline

SAMPLE_USER_INPUT = (
    "While the model starts, your PC can be slow or stop responding for 1-3 minutes "
    "(longest the first time). Strata loads 35-55 GB into your RAM and locks part of "
    "it for the graphics card. This is normal. Wait, and don't close the window. "
    "The window shows what Strata is doing."
)


def mock_llm_service(prompt_with_symbols: str) -> str:
    """Simulates an LLM receiving the symbolic prompt and emitting compressed symbols.
    
    Notice: The LLM outputs ONLY 33 tokens instead of 69 natural English tokens!
    """
    print("\n   [LLM REASONING ON COMPRESSED SYMBOLS...]")
    print(f"   LLM Received Prompt Length: {len(prompt_with_symbols)} chars")
    # LLM emits pure compressed symbols:
    return "Ω[M·PC·~0·1-3m·1st:max·S·35-55G·RAM·GPU·ok·win]"


def main():
    pipeline = NeuroSymbolicPipeline()
    
    print("\n" + "=" * 105)
    print(" " * 20 + "BIDIRECTIONAL NEURO-SYMBOLIC LLM PIPELINE DEMO")
    print("=" * 105)
    
    print("\n[STEP 1: USER INPUT TEXT (Human-Readable)]")
    print("-" * 105)
    print(f"\"{SAMPLE_USER_INPUT}\"")
    
    # Run the full turn
    result = pipeline.execute_turn(SAMPLE_USER_INPUT, llm_caller=mock_llm_service)
    
    in_stage = result["input_stage"]
    out_stage = result["output_stage"]
    metrics = result["total_token_metrics"]
    
    print("\n[STEP 2: PRE-PROCESSOR (Parser Compresses to Symbols -> Sent to LLM)]")
    print("-" * 105)
    print(f"Strategy Used:       {in_stage['mode_used']}")
    print(f"Compressed Symbol:   {in_stage['compressed_symbol']}")
    print(f"Original In Tokens:  {in_stage['original_tokens']} tokens")
    print(f"Symbolic In Tokens:  {in_stage['compressed_tokens']} tokens (Saved {in_stage['input_token_saving_pct']:.1f}% Prompt Tokens)")
    print(f"\nExact Prompt Sent to LLM:")
    print(f"  {in_stage['llm_prompt']}")
    
    print("\n[STEP 3: LLM GENERATION (LLM Emits Compressed Symbol Directly)]")
    print("-" * 105)
    print(f"LLM Generated Token Stream:  {result['llm_intermediate_reply']}")
    print(f"Generation Cost:             {out_stage['compressed_tokens']} tokens (vs {out_stage['human_tokens']} tokens for English prose)")
    print(f"Generation Token Saving:     {out_stage['output_token_saving_pct']:.1f}% Saved")
    
    print("\n[STEP 4: POST-PROCESSOR (Parser Expands Symbol -> Human-Readable Text)]")
    print("-" * 105)
    print(f"Strategy Used:       {out_stage['mode_used']}")
    print(f"Final Human Text Output Presented to User:\n")
    print(f"\"{out_stage['human_readable_output']}\"")
    
    print("\n" + "=" * 105)
    print(" " * 34 + "END-TO-END PIPELINE METRICS")
    print("=" * 105)
    print(f"Total Standard Tokens (Without Pipeline):  {metrics['total_standard_tokens']} tokens")
    print(f"Total Symbolic Tokens (With Pipeline):      {metrics['total_symbolic_tokens']} tokens")
    print(f"OVERALL TOKEN & BANDWIDTH REDUCTION:       {metrics['overall_token_saving_pct']:.1f}% SAVED")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
