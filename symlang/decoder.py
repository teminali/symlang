"""Decoder: expands Tier6AST states back into text and checks semantic fidelity.

Provides:
1. A real, total expansion function for ONE grammar (`expand_frame`: Tier6AST -> sentence) and its inverse
   (`parse_canonical_sentence`: sentence -> Tier6AST). `SymbolicDecoder.decode_deterministic` is built on them.
2. Universal LLM decompression prompts.
3. A keyword-presence check (`verify_semantic_fidelity`). It is a coarse smoke test, NOT proof of meaning: it cannot
   tell "do not close the window" from "you can close the window".

Scope of the frame: the 294,912 states of the 11-slot grammar in `meta_tier.DOMAIN_SLOTS`. Each slot value has exactly
one fixed phrase (FRAME_PHRASES), and the sentence is a fixed template around those phrases, so text -> AST -> text is
the identity ONLY for sentences that are exactly what `expand_frame` emits (whitespace-normalised). Free-form English
is not in the frame: `decode_deterministic` returns anything it cannot parse unchanged, and the pipeline's round-trip
guard (`pipeline.verify_roundtrip`) refuses to use the frame for any input that does not reproduce exactly.

The Tier6AST alone is NOT enough to reproduce arbitrary English (it holds 11 slot values, not wording). The frame makes
it exact by fixing the wording per slot value; that is the "shared codebook" the README talks about.
"""

import re
import string
from typing import Any, Dict, List, Optional, Tuple

from symlang.meta_tier import DOMAIN_SLOTS, MetaTierCodec, Tier6AST, Tier6Parser


def normalize_ws(text: str) -> str:
    """Collapse every whitespace run to one space and strip the ends (the equivalence used by round-trip checks)."""
    return " ".join(text.split())


# One fixed phrase per slot value. Phrases must be unique within a slot (that is what makes the map invertible);
# FRAME_PHRASES is checked against DOMAIN_SLOTS at import time and by tests/test_honest_pipeline.py.
# alloc_entity values carry (sentence-initial form, mid-sentence form).
FRAME_PHRASES: Dict[str, Dict[str, Any]] = {
    "trigger_entity": {"M": "the model starts", "SYS": "the system starts", "APP": "the app starts", "OS": "the OS starts"},
    "target_entity": {"PC": "PC", "HOST": "host", "CLIENT": "client"},
    "degraded_state": {"~0": "slow or stop responding", "~": "slow", "0": "unresponsive", "!": "locked up"},
    "duration": {"1-3m": "1-3 minutes", "1-5m": "1-5 minutes", "2-4m": "2-4 minutes", "<1m": "under a minute"},
    "condition": {
        "1st:max": "longest the first time",
        "max": "longest it can take",
        "init:max": "longest at initialisation",
        "cold": "longest from a cold start",
    },
    "alloc_entity": {"S": ("Strata", "Strata"), "SYS": ("The system", "the system")},
    "alloc_amount": {"35-55G": "35-55 GB", "16-32G": "16-32 GB", "8-16G": "8-16 GB", ">64G": "over 64 GB"},
    "target_hw1": {"RAM": "RAM", "VRAM": "VRAM"},
    "target_hw2": {"GPU": "graphics card", "CPU": "processor"},
    "assert_state": {"ok": "normal", "norm": "expected", "valid": "valid", "ready": "ready"},
    "window_entity": {"win": "window", "app": "app window", "ui": "interface"},
}

# The canonical sentence is expand_frame(<the sample state>); this is the exact string used in the README and demo.
FRAME_TEMPLATE = (
    "While {trigger_entity}, your {target_entity} can be {degraded_state} for {duration} ({condition}). "
    "{alloc_init} loads {alloc_amount} into your {target_hw1} and locks part of it for the {target_hw2}. "
    "This is {assert_state}. Wait, and don't close the {window_entity}. "
    "The {window_entity} shows what {alloc_mid} is doing."
)


def _check_phrase_tables() -> None:
    for slot, values in DOMAIN_SLOTS:
        table = FRAME_PHRASES[slot]
        if list(table) != list(values):
            raise AssertionError(f"FRAME_PHRASES[{slot}] keys {list(table)} != DOMAIN_SLOTS {values}")
        flat = list(table.values())
        if len(set(flat)) != len(flat):
            raise AssertionError(f"FRAME_PHRASES[{slot}] phrases are not unique: not invertible")


_check_phrase_tables()


def expand_frame(ast: Tier6AST) -> str:
    """Expand a Tier6AST into its one canonical sentence (total on in-domain states, deterministic).

    Raises ValueError for a slot value outside the closed domain: the AST then carries information the frame has no
    wording for, so it cannot be expanded exactly and we refuse instead of guessing.
    """
    parts: Dict[str, str] = {}
    for slot, _values in DOMAIN_SLOTS:
        value = getattr(ast, slot)
        if value not in FRAME_PHRASES[slot]:
            raise ValueError(f"'{value}' is not a value of slot '{slot}'; the frame has no phrase for it")
        phrase = FRAME_PHRASES[slot][value]
        if slot == "alloc_entity":
            parts["alloc_init"], parts["alloc_mid"] = phrase
        else:
            parts[slot] = phrase
    return FRAME_TEMPLATE.format(**parts)


def _alt(phrases: List[str]) -> str:
    return "|".join(re.escape(p) for p in sorted(set(phrases), key=len, reverse=True))


def _build_frame_regex() -> "re.Pattern[str]":
    """Regex generated from FRAME_TEMPLATE itself, so template and parser cannot drift apart."""
    out: List[str] = []
    seen = set()
    for literal, field, _spec, _conv in string.Formatter().parse(FRAME_TEMPLATE):
        out.append(re.escape(literal))
        if field is None:
            continue
        if field in seen:  # a slot used twice (window_entity) must repeat the same phrase
            out.append(f"(?P={field})")
            continue
        seen.add(field)
        if field == "alloc_init":
            phrases = [v[0] for v in FRAME_PHRASES["alloc_entity"].values()]
        elif field == "alloc_mid":
            phrases = [v[1] for v in FRAME_PHRASES["alloc_entity"].values()]
        else:
            phrases = list(FRAME_PHRASES[field].values())
        out.append(f"(?P<{field}>{_alt(phrases)})")
    return re.compile("^" + "".join(out) + "$")


_FRAME_RE = _build_frame_regex()


def parse_canonical_sentence(text: str) -> Optional[Tier6AST]:
    """Sentence -> Tier6AST, or None if `text` is not exactly a sentence this frame can emit.

    The match is followed by an expand_frame() comparison, so a returned AST always satisfies
    normalize_ws(expand_frame(ast)) == normalize_ws(text). That check, not the regex, is the guarantee.
    """
    norm = normalize_ws(text)
    m = _FRAME_RE.match(norm)
    if not m:
        return None
    vec = []
    for slot, _values in DOMAIN_SLOTS:
        key = "alloc_init" if slot == "alloc_entity" else slot
        shown = m.group(key)
        if slot == "alloc_entity":
            hits = [k for k, v in FRAME_PHRASES[slot].items() if v[0] == shown]
        else:
            hits = [k for k, v in FRAME_PHRASES[slot].items() if v == shown]
        if len(hits) != 1:
            return None
        vec.append(hits[0])
    ast = Tier6AST.from_vector(vec)
    return ast if expand_frame(ast) == norm else None


def ast_from_symbol(symbol: str) -> Optional[Tier6AST]:
    """Parse a Tier 6 operator string, a Layer 7 tensor or a Layer 8 radix vector into an AST; None if it is none of them."""
    s = symbol.strip()
    for parse in (Tier6Parser.parse, MetaTierCodec.decode_layer8_radix, MetaTierCodec.decode_layer7_tensor):
        try:
            return parse(s)
        except Exception:
            continue
    return None


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
        """Tier 6 / Layer 7 / Layer 8 symbol -> AST -> sentence, by `expand_frame`. Anything else is returned unchanged.

        There is no keyword lookup and no canned paragraph: the text is a function of the parsed AST only. A symbol
        that does not parse, or whose slot values are outside the frame's closed domain, comes back unchanged.
        """
        ast = ast_from_symbol(compressed_text)
        if ast is None:
            return compressed_text
        try:
            return expand_frame(ast)
        except ValueError:
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
