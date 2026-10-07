#!/usr/bin/env python3
"""Demo of the token-aware, round-trip-verified pipeline.

Shows: [Human Input Text] -> router -> [payload] -> LLM -> parser -> [Human-Readable Text].

HONESTY NOTE: the "LLM" below is a hardcoded stub, not a model; nothing here measures real model behaviour. The
numbers are token counts of the encodings (cl100k_base), and the compressed form is used only for the one canonical
sentence of the one pre-agreed grammar. Any other text is sent Direct, with 0% saving.
"""

import sys
from symlang.pipeline import NeuroSymbolicPipeline

SAMPLE_USER_INPUT = (
    "While the model starts, your PC can be slow or stop responding for 1-3 minutes "
    "(longest the first time). Strata loads 35-55 GB into your RAM and locks part of "
    "it for the graphics card. This is normal. Wait, and don't close the window. "
    "The window shows what Strata is doing."
)


def _tokens_vs(tokens: int, raw: int) -> str:
    """'+65.2% tokens vs raw' (costs more) or '-52.2% tokens vs raw' (costs less); never relabelled as a saving."""
    return f"{(tokens / raw - 1.0) * 100.0:+.1f}% tokens vs raw"


def mock_llm_service(prompt_with_symbols: str) -> str:
    """STUB, not a model: ignores the prompt and returns a fixed reply in the frame."""
    print("\n   [SIMULATED LLM: hardcoded reply, no model is called]")
    print(f"   LLM Received Prompt Length: {len(prompt_with_symbols)} chars")
    return "Ω[M·PC·~0·1-3m·1st:max·S·35-55G·RAM·GPU·ok·win]"


def main():
    pipeline = NeuroSymbolicPipeline()
    
    print("\n" + "=" * 105)
    print(" " * 14 + "TOKEN-AWARE ROUTER DEMO (SIMULATED LLM, ONE PRE-AGREED SENTENCE)")
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
    print(f"Symbolic In Tokens:  {in_stage['compressed_tokens']} tokens (payload saving {in_stage['input_token_saving_pct']:.1f}%, tokenizer {in_stage['token_counter']})")
    print(f"Whole prompt:        {in_stage['llm_prompt_tokens']} tokens incl. protocol line ({_tokens_vs(in_stage['llm_prompt_tokens'], in_stage['original_tokens'])}, one-shot)")
    print(f"Round trip verified: {in_stage['roundtrip_verified']} (exact: {in_stage['roundtrip_exact']})")
    print(f"\nExact Prompt Sent to LLM:")
    print(f"  {in_stage['llm_prompt']}")
    
    print("\n[STEP 3: SIMULATED LLM REPLY (hardcoded in this script; not a model's output)]")
    print("-" * 105)
    print(f"Hardcoded reply:             {result['llm_intermediate_reply']}")
    print(f"Reply tokens:                {out_stage['compressed_tokens']} (vs {out_stage['human_tokens']} for the same reply as English prose)")
    print(f"Token change on the reply:   {_tokens_vs(out_stage['compressed_tokens'], out_stage['human_tokens'])} (a property of this fixed reply, not a measurement of any model)")
    
    print("\n[STEP 4: POST-PROCESSOR (Parser Expands Symbol -> Human-Readable Text)]")
    print("-" * 105)
    print(f"Strategy Used:       {out_stage['mode_used']}")
    print(f"Final Human Text Output Presented to User:\n")
    print(f"\"{out_stage['human_readable_output']}\"")
    
    print("\n" + "=" * 105)
    print(" " * 34 + "END-TO-END PIPELINE METRICS")
    print("=" * 105)
    print(f"Raw text in + raw reply out:               {metrics['total_standard_tokens']} tokens")
    print(f"Payloads only (frame in + frame out):      {metrics['total_symbolic_tokens']} tokens ({metrics['overall_token_saving_pct']:.1f}% fewer; simulated LLM)")
    with_scaffold = metrics['total_symbolic_tokens_with_prompt_scaffold']
    print(f"Counting the prompt's protocol line:       {with_scaffold} tokens ({_tokens_vs(with_scaffold, metrics['total_standard_tokens'])}; "
          f"{'costs MORE than raw' if with_scaffold > metrics['total_standard_tokens'] else 'fewer than raw'} for a single one-shot message)")
    print("The frame pays off only if the protocol line is amortised (system prompt) or both ends already hold the codebook.")
    print("=" * 105)

    other = "Please wait; it is fine to close the window now."
    r = pipeline.prepare_llm_input(other)
    print(f"\nAny other text (\"{other}\"): mode = {r['mode_used']}, saving = {r['input_token_saving_pct']:.1f}%")
    print("Only the exact canonical sentence of the one grammar takes the frame; a near-miss, even one that reverses the meaning, goes Direct.")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
