"""Meta-Tier encodings for ONE pre-agreed grammar (294,912 states) and its exact parser.

Scope, stated plainly: every layer below encodes a state of a single fixed 11-slot grammar (see DOMAIN_SLOTS and
TOTAL_STATES). Because sender and receiver share that closed codebook, a state fits in 18.17 bits, so Layers 11 and 12
can be 3 ASCII characters or one Unicode code point. That is a property of the shared codebook, NOT general text
compression: arbitrary text is not in the grammar and cannot be encoded here. Character counts below are not token
counts; measure tokens with the target tokenizer (see benchmark_tokens.py).

Layers (character counts for the one sample sentence):
- Layer 7:  SymTensor (Algebraic State Tensor Ψ - 49 chars)
- Layer 8:  SymRadix (Canonical Position Vector Ω - 47 chars)
- Layer 9A: SymBase85 (Radix-85 stream of slot indices - 15 chars)
- Layer 9B: SymRune (16-bit rune packing of slot indices - 6 chars)
- Layer 10: SymGodel (integer N from the slot-index bytes)
- Layer 11: SymTriad (the state id in 3 ASCII characters)
- Layer 12: SymSingular (the state id as 1 Unicode code point)

What is verified: parse -> AST -> encode -> decode -> emit_tier6 is the identity for every in-domain state
(tests/test_meta_tier.py). Out-of-domain slot values and out-of-range state ids raise ValueError instead of being
silently remapped.
"""

import re
import base64
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any


# Discrete canonical domains defining the 11-dimensional AST state space Ω
DOMAIN_SLOTS = [
    ("trigger_entity", ["M", "SYS", "APP", "OS"]),
    ("target_entity",  ["PC", "HOST", "CLIENT"]),
    ("degraded_state", ["~0", "~", "0", "!"]),
    ("duration",       ["1-3m", "1-5m", "2-4m", "<1m"]),
    ("condition",      ["1st:max", "max", "init:max", "cold"]),
    ("alloc_entity",   ["S", "SYS"]),
    ("alloc_amount",   ["35-55G", "16-32G", "8-16G", ">64G"]),
    ("target_hw1",     ["RAM", "VRAM"]),
    ("target_hw2",     ["GPU", "CPU"]),
    ("assert_state",   ["ok", "norm", "valid", "ready"]),
    ("window_entity",  ["win", "app", "ui"]),
]

DOMAINS_MAP = {name: {val: i for i, val in enumerate(vals)} for name, vals in DOMAIN_SLOTS}

# Size of the one pre-agreed grammar: the product of the slot cardinalities (4*3*4*4*4*2*4*2*2*4*3 = 294,912).
TOTAL_STATES = 1
for _name, _vals in DOMAIN_SLOTS:
    TOTAL_STATES *= len(_vals)


@dataclass(frozen=True)
class Tier6AST:
    """Formal Abstract Syntax Tree (AST) for Tier 6 SymOperator statements."""
    trigger_entity: str   # e.g., 'M'
    target_entity: str    # e.g., 'PC'
    degraded_state: str   # e.g., '~0'
    duration: str         # e.g., '1-3m'
    condition: str        # e.g., '1st:max'
    alloc_entity: str     # e.g., 'S'
    alloc_amount: str     # e.g., '35-55G'
    target_hw1: str       # e.g., 'RAM'
    target_hw2: str       # e.g., 'GPU'
    assert_state: str     # e.g., 'ok'
    window_entity: str    # e.g., 'win'

    def emit_tier6(self) -> str:
        """Deterministic synthesizer emitting the exact Tier 6 representation."""
        return (
            f"▶{self.trigger_entity}⇒{self.target_entity}{self.degraded_state} "
            f"{self.duration}({self.condition})."
            f"{self.alloc_entity}:{self.alloc_amount}→"
            f"{self.target_hw1}+{self.target_hw2}."
            f"✓{self.assert_state}."
            f"⏳¬✕{self.window_entity}:{self.window_entity}=👁{self.alloc_entity}"
        )

    def to_vector(self) -> List[str]:
        """Convert AST into ordered positional coordinate vector."""
        return [
            self.trigger_entity,
            self.target_entity,
            self.degraded_state,
            self.duration,
            self.condition,
            self.alloc_entity,
            self.alloc_amount,
            self.target_hw1,
            self.target_hw2,
            self.assert_state,
            self.window_entity,
        ]

    @classmethod
    def from_vector(cls, v: List[str]) -> "Tier6AST":
        """Reconstruct AST from ordered coordinate vector."""
        if len(v) != 11:
            raise ValueError(f"Expected 11 vector components, got {len(v)}")
        return cls(
            trigger_entity=v[0],
            target_entity=v[1],
            degraded_state=v[2],
            duration=v[3],
            condition=v[4],
            alloc_entity=v[5],
            alloc_amount=v[6],
            target_hw1=v[7],
            target_hw2=v[8],
            assert_state=v[9],
            window_entity=v[10],
        )


class Tier6Parser:
    """Formal deterministic parser for Tier 6 strings."""

    GRAMMAR_REGEX = re.compile(
        r"^▶(?P<m>[A-Za-z0-9_]+)⇒(?P<pc>[A-Za-z0-9_]+)(?P<deg>[~0!]+)\s+"
        r"(?P<dur>[0-9a-zA-Z\-<]+)\((?P<cond>[^)]+)\)\."
        r"(?P<s>[A-Za-z0-9_]+):(?P<alloc>[0-9a-zA-Z\-<>]+)→"
        r"(?P<t1>[A-Za-z0-9_]+)\+(?P<t2>[A-Za-z0-9_]+)\."
        r"✓(?P<ok>[A-Za-z0-9_]+)\."
        r"⏳¬✕(?P<w>[A-Za-z0-9_]+):(?P=w)=👁(?P=s)$"
    )

    @classmethod
    def parse(cls, tier6_text: str) -> Tier6AST:
        """Parses a Tier 6 string into a strongly-typed Tier6AST."""
        match = cls.GRAMMAR_REGEX.match(tier6_text.strip())
        if not match:
            raise ValueError(
                f"String does not conform to Tier 6 formal grammar: '{tier6_text}'"
            )
        d = match.groupdict()
        return Tier6AST(
            trigger_entity=d["m"],
            target_entity=d["pc"],
            degraded_state=d["deg"],
            duration=d["dur"],
            condition=d["cond"],
            alloc_entity=d["s"],
            alloc_amount=d["alloc"],
            target_hw1=d["t1"],
            target_hw2=d["t2"],
            assert_state=d["ok"],
            window_entity=d["w"],
        )


class MetaTierCodec:
    """Mathematical multi-layer encoder and decoder that operates on Tier 6."""

    # -------------------------------------------------------------
    # Layer 7: Algebraic State Tensor (Ψ-Form)
    # -------------------------------------------------------------
    @staticmethod
    def encode_layer7_tensor(ast: Tier6AST) -> str:
        return (
            f"Ψ⟨{ast.trigger_entity}⇒{ast.target_entity}{ast.degraded_state}@"
            f"{ast.duration}*{ast.condition};"
            f"{ast.alloc_entity}:{ast.alloc_amount}→{ast.target_hw1}+{ast.target_hw2};"
            f"{ast.assert_state};{ast.window_entity}=👁{ast.alloc_entity}⟩"
        )

    @staticmethod
    def decode_layer7_tensor(tensor_str: str) -> Tier6AST:
        pattern = re.compile(
            r"^Ψ⟨(?P<m>[A-Za-z0-9_]+)⇒(?P<pc>[A-Za-z0-9_]+)(?P<deg>[~0!]+)@"
            r"(?P<dur>[0-9a-zA-Z\-<]+)\*(?P<cond>[^;]+);"
            r"(?P<s>[A-Za-z0-9_]+):(?P<alloc>[0-9a-zA-Z\-<>]+)→(?P<t1>[A-Za-z0-9_]+)\+(?P<t2>[A-Za-z0-9_]+);"
            r"(?P<ok>[A-Za-z0-9_]+);(?P<w>[A-Za-z0-9_]+)=👁(?P=s)⟩$"
        )
        match = pattern.match(tensor_str.strip())
        if not match:
            raise ValueError(f"Invalid Layer 7 tensor string: {tensor_str}")
        d = match.groupdict()
        return Tier6AST(
            trigger_entity=d["m"],
            target_entity=d["pc"],
            degraded_state=d["deg"],
            duration=d["dur"],
            condition=d["cond"],
            alloc_entity=d["s"],
            alloc_amount=d["alloc"],
            target_hw1=d["t1"],
            target_hw2=d["t2"],
            assert_state=d["ok"],
            window_entity=d["w"],
        )

    # -------------------------------------------------------------
    # Layer 8: Canonical Radix Vector (Ω-Form)
    # -------------------------------------------------------------
    @staticmethod
    def encode_layer8_radix(ast: Tier6AST) -> str:
        coords = "·".join(ast.to_vector())
        return f"Ω[{coords}]"

    @staticmethod
    def decode_layer8_radix(radix_str: str) -> Tier6AST:
        if not (radix_str.startswith("Ω[") and radix_str.endswith("]")):
            raise ValueError(f"Invalid Layer 8 radix string format: {radix_str}")
        inner = radix_str[2:-1]
        parts = inner.split("·")
        return Tier6AST.from_vector(parts)

    # -------------------------------------------------------------
    # Layer 9A: Base85 Coordinate Index Stream
    # -------------------------------------------------------------
    @staticmethod
    def encode_layer9a_base85(ast: Tier6AST) -> str:
        raw_bytes = MetaTierCodec._ast_to_index_bytes(ast)
        return "§" + base64.b85encode(raw_bytes).decode("ascii")

    @staticmethod
    def decode_layer9a_base85(b85_str: str) -> Tier6AST:
        if b85_str.startswith("§"):
            b85_str = b85_str[1:]
        raw_bytes = base64.b85decode(b85_str.encode("ascii"))
        return MetaTierCodec._index_bytes_to_ast(raw_bytes)

    # -------------------------------------------------------------
    # Layer 9B: 16-bit Mathematical Rune Packing (6 chars)
    # -------------------------------------------------------------
    @staticmethod
    def encode_layer9b_runes(ast: Tier6AST) -> str:
        raw_bytes = bytearray(MetaTierCodec._ast_to_index_bytes(ast))
        if len(raw_bytes) % 2 != 0:
            raw_bytes.append(0xFF)
        runes = []
        for i in range(0, len(raw_bytes), 2):
            val = (raw_bytes[i] << 8) | raw_bytes[i + 1]
            runes.append(chr(0x4E00 + val))
        return "".join(runes)

    @staticmethod
    def decode_layer9b_runes(rune_str: str) -> Tier6AST:
        unpacked = bytearray()
        for r in rune_str:
            val = ord(r) - 0x4E00
            unpacked.append((val >> 8) & 0xFF)
            low = val & 0xFF
            if low != 0xFF:
                unpacked.append(low)
        return MetaTierCodec._index_bytes_to_ast(bytes(unpacked))

    # -------------------------------------------------------------
    # Layer 10: Bijective BigInteger N ∈ ℕ
    # -------------------------------------------------------------
    @staticmethod
    def encode_layer10_godel(ast: Tier6AST) -> int:
        raw_bytes = MetaTierCodec._ast_to_index_bytes(ast)
        val = 1
        for b in raw_bytes:
            val = (val << 8) | b
        return val

    @staticmethod
    def decode_layer10_godel(godel_int: int) -> Tier6AST:
        b_list = []
        val = godel_int
        while val > 1:
            b_list.append(val & 0xFF)
            val >>= 8
        raw_bytes = bytes(reversed(b_list))
        return MetaTierCodec._index_bytes_to_ast(raw_bytes)

    # -------------------------------------------------------------
    # Layer 11: SymTriad (the state id as 3 ASCII characters; needs the shared codebook)
    # -------------------------------------------------------------
    @staticmethod
    def encode_layer11_triad(ast: Tier6AST) -> str:
        """Packs the 18.17-bit state integer into exactly 3 ASCII characters (Base-85)."""
        state_id = MetaTierCodec._ast_to_state_id(ast)
        # Convert state_id ∈ [0, 294911] into 3 Base-85 characters (85^3 = 614,125)
        chars = []
        val = state_id
        for _ in range(3):
            chars.append(chr(33 + (val % 85)))
            val //= 85
        return "".join(reversed(chars))

    @staticmethod
    def decode_layer11_triad(triad_str: str) -> Tier6AST:
        """Inverts the canonical 3-character triad back into a Tier6AST.

        Strict on purpose: Layer 11 is a bijection between the 294,912 states and their codes ("one state, one code"),
        so only the canonical spelling is accepted. Each character must be in '!'..'u' (the 85 digits) and the value
        must be a state id in [0, TOTAL_STATES). Anything else (wrong length, a character outside '!'..'u' such as the
        alias '!v!', an id past the last state) raises ValueError instead of being silently mapped onto some state.
        """
        if len(triad_str) != 3:
            raise ValueError(f"Expected 3 ASCII characters, got {len(triad_str)}")
        val = 0
        for c in triad_str:
            digit = ord(c) - 33
            if not 0 <= digit < 85:
                raise ValueError(
                    f"{triad_str!r} is not a canonical triad: {c!r} is outside '!'..'u'; each of the 3 characters "
                    f"must be one of the 85 digits '!'..'u' (a decoder that accepted other spellings would map "
                    f"several codes onto one state)"
                )
            val = val * 85 + digit
        return MetaTierCodec._state_id_to_ast(val)  # ValueError if val >= TOTAL_STATES

    # -------------------------------------------------------------
    # Layer 12: SymSingular (the state id as 1 code point; needs the shared codebook)
    # -------------------------------------------------------------
    @staticmethod
    def encode_layer12_singular(ast: Tier6AST) -> str:
        """Packs the entire sentence into a SINGLE Unicode code point (Plane 1)."""
        state_id = MetaTierCodec._ast_to_state_id(ast)
        # Map state_id ∈ [0, 294911] to Unicode Plane 1 (U+10000 to U+57FFF)
        return chr(0x10000 + state_id)

    @staticmethod
    def decode_layer12_singular(char: str) -> Tier6AST:
        """Inverts 1 single Unicode character back into Tier6AST."""
        if len(char) != 1:
            raise ValueError(f"Expected 1 single character, got {len(char)}")
        state_id = ord(char) - 0x10000
        return MetaTierCodec._state_id_to_ast(state_id)

    # -------------------------------------------------------------
    # Internal Coordinate Space Math Helpers
    # -------------------------------------------------------------
    @staticmethod
    def _ast_to_state_id(ast: Tier6AST) -> int:
        """Calculates exact integer state coordinate in mixed-radix space Ω."""
        vec = ast.to_vector()
        state_id = 0
        multiplier = 1
        for (slot_name, values), term in zip(DOMAIN_SLOTS, vec):
            # A term outside the closed domain has no coordinate in the 294,912-state space.
            # (It used to be silently mapped to index 0, which made Layers 11/12 lossy without saying so.)
            if term not in values:
                raise ValueError(
                    f"'{term}' is not in the closed domain of slot '{slot_name}' {values}; "
                    f"it has no Layer 11/12 state id"
                )
            state_id += values.index(term) * multiplier
            multiplier *= len(values)
        return state_id

    @staticmethod
    def _state_id_to_ast(state_id: int) -> Tier6AST:
        """Reconstructs AST from mixed-radix state integer."""
        if not (0 <= state_id < TOTAL_STATES):
            # Without this check ids >= TOTAL_STATES silently alias onto valid states.
            raise ValueError(f"state id {state_id} is outside [0, {TOTAL_STATES}); not a state of this grammar")
        vec = []
        rem = state_id
        for slot_name, values in DOMAIN_SLOTS:
            card = len(values)
            idx = rem % card
            vec.append(values[idx])
            rem //= card
        return Tier6AST.from_vector(vec)

    @staticmethod
    def _ast_to_index_bytes(ast: Tier6AST) -> bytes:
        vec = ast.to_vector()
        out = bytearray()
        for (slot_name, values), term in zip(DOMAIN_SLOTS, vec):
            if term in values:
                out.append(values.index(term))
            else:
                raw = term.encode("utf-8")
                out.append(0x80 | (len(raw) & 0x7F))
                out.extend(raw)
        return bytes(out)

    @staticmethod
    def _index_bytes_to_ast(raw_bytes: bytes) -> Tier6AST:
        vec = []
        i = 0
        slot_idx = 0
        while i < len(raw_bytes) and slot_idx < len(DOMAIN_SLOTS):
            slot_name, values = DOMAIN_SLOTS[slot_idx]
            head = raw_bytes[i]
            i += 1
            if head < 0x80:
                if head < len(values):
                    vec.append(values[head])
                else:
                    raise ValueError(f"index {head} is outside slot '{slot_name}' ({len(values)} values)")
            else:
                length = head & 0x7F
                term = raw_bytes[i : i + length].decode("utf-8")
                i += length
                vec.append(term)
            slot_idx += 1
        return Tier6AST.from_vector(vec)


def verify_mathematical_roundtrip(tier6_input: str) -> Dict[str, Any]:
    """Checks that Layers 7, 8, 11 and 12 reproduce `tier6_input` exactly (one grammar; says nothing about other text)."""
    ast_orig = Tier6Parser.parse(tier6_input)
    
    l7 = MetaTierCodec.encode_layer7_tensor(ast_orig)
    tier6_l7 = MetaTierCodec.decode_layer7_tensor(l7).emit_tier6()
    
    l8 = MetaTierCodec.encode_layer8_radix(ast_orig)
    tier6_l8 = MetaTierCodec.decode_layer8_radix(l8).emit_tier6()
    
    l11 = MetaTierCodec.encode_layer11_triad(ast_orig)
    tier6_l11 = MetaTierCodec.decode_layer11_triad(l11).emit_tier6()
    
    l12 = MetaTierCodec.encode_layer12_singular(ast_orig)
    tier6_l12 = MetaTierCodec.decode_layer12_singular(l12).emit_tier6()
    
    all_exact = (
        tier6_l7 == tier6_input and
        tier6_l8 == tier6_input and
        tier6_l11 == tier6_input and
        tier6_l12 == tier6_input
    )
    
    return {
        "tier6_original": tier6_input,
        "ast": asdict(ast_orig),
        "layers": {
            "Layer 7 (SymTensor)": {"encoded": l7, "verified": tier6_l7 == tier6_input},
            "Layer 8 (SymRadix)": {"encoded": l8, "verified": tier6_l8 == tier6_input},
            "Layer 11 (SymTriad)": {"encoded": l11, "verified": tier6_l11 == tier6_input},
            "Layer 12 (SymSingular)": {"encoded": l12, "verified": tier6_l12 == tier6_input},
        },
        "all_layers_mathematically_proven": all_exact,
    }

