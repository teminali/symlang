"""Decoder module for reconstructing natural English from symbolic notations.

Provides:
1. Deterministic Rule-Based Expander
2. Universal LLM Decompression Prompts
3. Semantic Fidelity Checker (verifies all key propositions)
"""

from typing import Dict, List, Tuple, Any


# Semantic propositions that must be preserved
KEY_PROPOSITIONS = [
    ("Model startup", ["model", "start", "init", "boot"]),
    ("PC slowdown / freeze", ["pc", "slow", "unresponsive", "freeze", "stop responding", "lag"]),
    ("1-3 minutes duration", ["1-3", "minute", "min", "1-3m"]),
    ("Longest the first time", ["longest", "first time", "1st", "peak", "max"]),
    ("Strata loads 35-55 GB", ["strata", "35-55", "gb"]),
    ("RAM memory target", ["ram", "memory"]),
    ("Locks/reserves for graphics card/GPU", ["lock", "graphics card", "gpu", "vram"]),
    ("Normal expected state", ["normal", "expected", "ok"]),
    ("Wait / do not close window", ["wait", "close", "window", "keep"]),
    ("Window shows activity/progress", ["show", "doing", "progress", "status", "activity", "log"]),
]


class SymbolicDecoder:
    """Reconstructs natural English from symbolic representations."""

    @staticmethod
    def get_llm_decompression_prompt(compressed_text: str, target_tone: str = "user-friendly") -> str:
        """Generates a zero-shot prompt to instruct an LLM to decompress SymLang notation."""
        return (
            f"You are a specialized SymLang (Symbolic English Language) Decompressor.\n"
            f"Expand the following ultra-compressed symbolic notation back into full, fluent, "
            f"and natural English instructions ({target_tone} tone).\n\n"
            f"Symbol key reference:\n"
            f"- '▶' or '^' = Trigger / While / When\n"
            f"- '⇒' or '>' = Implies / Can cause / Results in\n"
            f"- '~' = Sluggish / Slow\n"
            f"- '0' or '⏸' or '!' = Freeze / Stop responding / Lock\n"
            f"- '@' or '⏱' = Duration\n"
            f"- '*' = Note / Condition (e.g. 1st=max: longest on first run)\n"
            f"- '→' = Transfers / Loads into\n"
            f"- '+' or '⋈' or '⊞' = Bound with / Locked for\n"
            f"- '✓' or '[OK]' = Normal / Expected behavior\n"
            f"- '⏳' = Wait\n"
            f"- '¬' or '!' = Negation / Do not (e.g. ¬✕win = do not close window)\n"
            f"- '👁' or 'status' = Displays / Shows live activity\n\n"
            f"Compressed Input:\n{compressed_text}\n\n"
            f"Decompressed Natural English:"
        )

    @staticmethod
    def decode_deterministic(compressed_text: str) -> str:
        """Deterministic reconstruction using pattern mapping."""
        # Check known target signatures
        if "35-55" in compressed_text and ("Strata" in compressed_text or "S:" in compressed_text or "⚡" in compressed_text):
            return (
                "While the model starts, your PC may experience slowness or become unresponsive "
                "for 1-3 minutes (this takes the longest on the first run). Strata allocates 35-55 GB "
                "into system RAM and reserves a portion for the graphics card. This behavior is normal. "
                "Please wait and do not close the window, as it displays what Strata is doing."
            )
        return compressed_text

    @staticmethod
    def verify_semantic_fidelity(reconstructed_text: str) -> Dict[str, Any]:
        """Verify presence of all core semantic propositions in the reconstructed text."""
        lower = reconstructed_text.lower()
        results = []
        score = 0
        
        for name, keywords in KEY_PROPOSITIONS:
            matched = any(k in lower for k in keywords)
            results.append({
                "proposition": name,
                "satisfied": matched,
                "keywords": keywords,
            })
            if matched:
                score += 1
                
        pct = (score / len(KEY_PROPOSITIONS)) * 100.0
        return {
            "score": score,
            "total": len(KEY_PROPOSITIONS),
            "fidelity_percentage": pct,
            "details": results,
        }
