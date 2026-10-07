"""Neuro-Symbolic LLM Bridge.

Connects ultra-compressed Layer 9B Runes with LLM instruction prompts
and deterministically parses LLM outputs back into validated Runes.
"""

import re
from typing import Tuple, Optional
from symlang.meta_tier import Tier6Parser, MetaTierCodec, Tier6AST


class LLMBridge:
    """Bridges LLMs with the deterministic SymLang parser."""

    @staticmethod
    def runes_to_llm_prompt(runes: str, task: str = "Explain to the user") -> str:
        """Unpacks 6 Runes (Layer 9B) deterministically into an LLM-optimal prompt."""
        ast = MetaTierCodec.decode_layer9b_runes(runes)
        tier6_text = ast.emit_tier6()
        radix_text = MetaTierCodec.encode_layer8_radix(ast)
        
        return (
            f"System State (SymLang Tier 6): {tier6_text}\n"
            f"Compact Frame (Layer 8): {radix_text}\n\n"
            f"Instruction: {task} in clear, friendly English."
        )

    @staticmethod
    def get_llm_generation_system_prompt() -> str:
        """System prompt instructing an LLM to respond in valid Tier 6 or Layer 8 format."""
        return (
            "You are a system monitor. Output system status in SymLang Layer 8 Radix format:\n"
            "Format: Ω[trigger·target·deg_state·duration·condition·allocator·amount·hw1·hw2·assertion·window]\n"
            "Example: Ω[M·PC·~0·1-3m·1st:max·S·35-55G·RAM·GPU·ok·win]\n"
            "Do not add conversational fluff. Output the Ω[...] block directly."
        )

    @staticmethod
    def parse_llm_response(llm_output: str) -> Tuple[Tier6AST, str]:
        """Deterministically extracts and parses LLM output back into AST and Layer 9B Runes.
        
        Supports:
        - Layer 8 Radix: Ω[...]
        - Layer 7 Tensor: Ψ⟨...⟩
        - Tier 6 Operator: ▶M⇒...
        """
        # Try Layer 8 Radix
        m_radix = re.search(r"Ω\[[^\]]+\]", llm_output)
        if m_radix:
            ast = MetaTierCodec.decode_layer8_radix(m_radix.group(0))
            runes = MetaTierCodec.encode_layer9b_runes(ast)
            return ast, runes

        # Try Layer 7 Tensor
        m_tensor = re.search(r"Ψ⟨[^⟩]+⟩", llm_output)
        if m_tensor:
            ast = MetaTierCodec.decode_layer7_tensor(m_tensor.group(0))
            runes = MetaTierCodec.encode_layer9b_runes(ast)
            return ast, runes

        # Try Tier 6 Operator
        m_tier6 = re.search(r"▶[^\n\r]+", llm_output)
        if m_tier6:
            ast = Tier6Parser.parse(m_tier6.group(0))
            runes = MetaTierCodec.encode_layer9b_runes(ast)
            return ast, runes

        raise ValueError(f"Could not extract valid SymLang notation from LLM output: {llm_output[:100]}...")
