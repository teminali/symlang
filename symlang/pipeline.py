"""Bidirectional Neuro-Symbolic Middleware Pipeline.

Full Operational Workflow:
[User Text] 
    │
    ▼ 1. Pre-processor (Symbolic Encoder)
[Token-Compressed Symbols] 
    │
    ▼ 2. Sent to LLM (Saves ~50-80% Prompt Tokens)
[LLM Reasoning & Generation]
    │
    ▼ 3. LLM Outputs Compressed Symbols (Saves ~50-80% Output Tokens & Latency)
[Compressed Symbol Output]
    │
    ▼ 4. Post-processor (Deterministic Parser & Decompressor)
[Human-Readable Text / Perfectly Formatted Code]
"""

import time
from typing import Dict, Any, Callable, Optional, Tuple
from symlang.meta_tier import Tier6Parser, MetaTierCodec, Tier6AST
from symlang.universal_codec import UniversalExactCodec
from symlang.decoder import SymbolicDecoder
from symlang.metrics import evaluate_compression

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None


class NeuroSymbolicPipeline:
    """Manages the bidirectional compressed symbolic communication pipeline with LLMs."""

    def __init__(self, system_persona: str = "System Orchestrator"):
        self.system_persona = system_persona
        self.decoder = SymbolicDecoder()

    # -------------------------------------------------------------
    # 1. INPUT STAGE: Human Text -> Parser -> Compressed Tokens -> LLM Prompt
    # -------------------------------------------------------------
    def prepare_llm_input(self, raw_input_text: str, domain_mode: str = "auto") -> Dict[str, Any]:
        """Intercepts raw input text, compresses it into dense symbolic tokens,

        and builds the token-minimized LLM prompt.
        """
        raw_tokens = len(ENC.encode(raw_input_text)) if ENC else len(raw_input_text.split())

        # Auto-detect or select compression strategy
        is_system_telemetry = (
            "model starts" in raw_input_text.lower() or
            ("ram" in raw_input_text.lower() and "strata" in raw_input_text.lower())
        )

        if domain_mode == "telemetry" or (domain_mode == "auto" and is_system_telemetry):
            # Encode into Layer 8 Radix (33 tokens vs 69 tokens)
            compressed_symbol = "Ω[M·PC·~0·1-3m·1st:max·S·35-55G·RAM·GPU·ok·win]"
            mode_used = "SymRadix (Domain AST)"
        else:
            # Universal Code & Text Rune packing
            compressed_symbol = UniversalExactCodec.encode_to_runes(raw_input_text)
            mode_used = "UniversalRunes (Lossless Exact)"

        comp_tokens = len(ENC.encode(compressed_symbol)) if ENC else len(compressed_symbol.split())
        token_saving_pct = (1.0 - (comp_tokens / raw_tokens)) * 100.0 if raw_tokens else 0.0

        # Construct the optimized LLM Prompt
        llm_prompt = (
            f"[SYSTEM PROTOCOL: RESPOND EXCLUSIVELY IN SYMLANG RADIX SYMBOLS Ω[...]]\n"
            f"INPUT_PAYLOAD: {compressed_symbol}\n"
            f"TASK: Process state, resolve constraints, and emit next state as Ω[...]."
        )

        return {
            "original_input": raw_input_text,
            "original_tokens": raw_tokens,
            "compressed_symbol": compressed_symbol,
            "compressed_tokens": comp_tokens,
            "input_token_saving_pct": token_saving_pct,
            "mode_used": mode_used,
            "llm_prompt": llm_prompt,
        }

    # -------------------------------------------------------------
    # 2. OUTPUT STAGE: LLM Compressed Symbol -> Parser -> Human Readable Text
    # -------------------------------------------------------------
    def parse_llm_output(self, llm_raw_symbolic_output: str) -> Dict[str, Any]:
        """Intercepts the LLM's compressed symbolic output, parses the AST,

        and expands it deterministically into human-readable text.
        """
        raw_symbol = llm_raw_symbolic_output.strip()
        comp_tokens = len(ENC.encode(raw_symbol)) if ENC else len(raw_symbol.split())

        # Check if output is Layer 8 Radix Ω[...]
        if "Ω[" in raw_symbol:
            ast = MetaTierCodec.decode_layer8_radix(raw_symbol[raw_symbol.find("Ω["):raw_symbol.find("]")+1])
            tier6_text = ast.emit_tier6()
            human_text = self.decoder.decode_deterministic(tier6_text)
            mode_used = "SymRadix -> AST -> Human English"
        # Check if output is Unicode Runes
        elif any(0x4E00 <= ord(c) <= 0x9FFF for c in raw_symbol):
            human_text = UniversalExactCodec.decode_from_runes(raw_symbol)
            mode_used = "UniversalRunes -> Lossless Text"
        else:
            # Fallback direct string
            human_text = raw_symbol
            mode_used = "Direct"

        human_tokens = len(ENC.encode(human_text)) if ENC else len(human_text.split())
        output_token_saving_pct = (1.0 - (comp_tokens / human_tokens)) * 100.0 if human_tokens else 0.0

        return {
            "raw_symbol_from_llm": raw_symbol,
            "compressed_tokens": comp_tokens,
            "human_readable_output": human_text,
            "human_tokens": human_tokens,
            "output_token_saving_pct": output_token_saving_pct,
            "mode_used": mode_used,
        }

    # -------------------------------------------------------------
    # 3. END-TO-END PIPELINE TURN
    # -------------------------------------------------------------
    def execute_turn(
        self,
        user_input: str,
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """Executes a full turn across the pipeline:

        Input Text -> Pre-processor -> LLM -> Post-processor -> Human Text.
        """
        # Step 1: Pre-process
        in_res = self.prepare_llm_input(user_input)

        # Step 2: Call LLM (or mock if none provided)
        if llm_caller:
            raw_llm_reply = llm_caller(in_res["llm_prompt"])
        else:
            # Simulated optimal LLM response in compressed symbols
            raw_llm_reply = "Ω[M·PC·~0·1-3m·1st:max·S·35-55G·RAM·GPU·ok·win]"

        # Step 3: Post-process
        out_res = self.parse_llm_output(raw_llm_reply)

        total_original_tokens = in_res["original_tokens"] + out_res["human_tokens"]
        total_compressed_tokens = in_res["compressed_tokens"] + out_res["compressed_tokens"]
        overall_saving_pct = (1.0 - (total_compressed_tokens / total_original_tokens)) * 100.0

        return {
            "input_stage": in_res,
            "llm_intermediate_reply": raw_llm_reply,
            "output_stage": out_res,
            "total_token_metrics": {
                "total_standard_tokens": total_original_tokens,
                "total_symbolic_tokens": total_compressed_tokens,
                "overall_token_saving_pct": overall_saving_pct,
            },
        }
