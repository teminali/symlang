"""Neuro-symbolic middleware pipeline: a token-aware router with a round-trip guard.

What this module does, and does not do:

    [User text] -> prepare_llm_input -> [payload] -> (LLM) -> parse_llm_output -> [text]

* The router builds every candidate representation of the input, counts tokens with the configured tokenizer
  (cl100k_base through tiktoken by default, any `token_counter` can be injected) and picks the candidate with the FEWEST
  tokens. The raw text is always a candidate, so the chosen payload never has more tokens than the raw text and the
  reported saving is never negative. If nothing beats the raw text the mode is 'Direct (no saving found)' and the saving
  is 0%.
* A non-raw candidate is eligible only if `verify_roundtrip` shows that expanding it reproduces the input. There is no
  keyword trigger. The domain frame (see symlang.decoder) applies only when the input IS the canonical sentence of the
  one pre-agreed 294,912-sentence grammar (whitespace-normalised); any other text never takes that path.
* zlib+runes and zlib+base85 are lossless but opaque: an LLM cannot inflate DEFLATE, and they typically cost 1.8-4x MORE
  tokens than the raw text. They are always measured and listed in `candidates`, but by default (`llm_readable_only=True`)
  they are not eligible for an LLM prompt. Pass `llm_readable_only=False` for storage/transport use between your own
  systems. The 3-character and 1-glyph frame codes are likewise codebook-only (the receiver must hold the codebook).
* The savings reported are payload savings. `llm_prompt_tokens` and `prompt_saving_pct` include the protocol scaffold, which
  for a single short message can erase the saving; a persistent system prompt amortises it.
* With no `llm_caller`, `execute_turn` runs a simulated LLM that echoes the payload (identity) and the result carries
  `simulated_llm: True`. Nothing about a real model's behaviour is measured in that case.
"""

import re
from typing import Any, Callable, Dict, List, Optional

from symlang.decoder import (
    SymbolicDecoder,
    expand_frame,
    normalize_ws,
    parse_canonical_sentence,
)
from symlang.meta_tier import MetaTierCodec
from symlang.metrics import get_token_counter
from symlang.universal_codec import UniversalExactCodec

try:  # kept for backwards compatibility; the pipeline itself goes through get_token_counter
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None

TokenCounter = Callable[[str], int]

_RADIX_RE = re.compile(r"Ω\[[^\]]*\]")
_FRAME_KEY = "Ω[trigger·target·degraded·duration·condition·allocator·amount·hw1·hw2·assertion·window]"


def verify_roundtrip(
    text: str,
    encoded: str,
    decoder: Callable[[str], str],
    normalize: Optional[Callable[[str], str]] = normalize_ws,
) -> bool:
    """True iff `decoder(encoded)` reproduces `text`.

    The comparison is after `normalize` (whitespace normalisation by default); pass `normalize=None` for an exact,
    byte-for-byte comparison (used for the lossless zlib codecs). Any exception raised by the decoder means the round
    trip failed: this function never raises and never returns True on a decode error.
    """
    try:
        back = decoder(encoded)
    except Exception:
        return False
    if not isinstance(back, str):
        return False
    if normalize is None:
        return back == text
    return normalize(back) == normalize(text)


def _saving_pct(compressed: int, original: int) -> float:
    return (1.0 - compressed / original) * 100.0 if original else 0.0


# Router tie-break order: when two candidates cost the same, the earlier one wins (raw is always first, so a tie
# with raw is never a "saving").
_PREFERENCE = ["direct", "frame_radix", "frame_triad", "frame_singular", "base85", "runes"]

_MODE_NAMES = {
    "direct": "Direct (no saving found)",
    "direct_forced": "Direct (compression disabled by domain_mode)",
    "frame_radix": "Frame/Radix (canonical sentence, round trip verified)",
    "frame_triad": "Frame/Triad (codebook-only, canonical sentence, round trip verified)",
    "frame_singular": "Frame/Singular (codebook-only, canonical sentence, round trip verified)",
    "base85": "Opaque/Base85 (zlib, lossless, not LLM-readable, round trip verified)",
    "runes": "Opaque/Runes (zlib, lossless, not LLM-readable, round trip verified)",
}

# Whether an LLM can act on the payload given only the protocol line in the prompt.
_LLM_READABLE = {"direct": True, "frame_radix": True}


class NeuroSymbolicPipeline:
    """Token-aware, round-trip-verified middleware between a user and an LLM."""

    def __init__(
        self,
        system_persona: str = "System Orchestrator",
        token_counter: Optional[TokenCounter] = None,
        encoding: str = "cl100k_base",
    ):
        self.system_persona = system_persona
        self.decoder = SymbolicDecoder()
        if token_counter is not None:
            self.token_counter_name = getattr(token_counter, "__name__", "custom")
            self.count_tokens: TokenCounter = token_counter
        else:
            self.token_counter_name, self.count_tokens = get_token_counter(encoding)

    # -------------------------------------------------------------
    # Candidate construction
    # -------------------------------------------------------------
    def _candidates(self, text: str, use_frame: bool) -> List[Dict[str, Any]]:
        """Every representation of `text`, with measured tokens and a round-trip verdict. Raw is always first."""
        count = self.count_tokens
        cands: List[Dict[str, Any]] = [{
            "key": "direct", "payload": text, "roundtrip_verified": True, "roundtrip_exact": True,
        }]

        if use_frame:
            ast = parse_canonical_sentence(text)
            if ast is not None:
                frames = {
                    "frame_radix": (MetaTierCodec.encode_layer8_radix(ast), MetaTierCodec.decode_layer8_radix),
                    "frame_triad": (MetaTierCodec.encode_layer11_triad(ast), MetaTierCodec.decode_layer11_triad),
                    "frame_singular": (MetaTierCodec.encode_layer12_singular(ast), MetaTierCodec.decode_layer12_singular),
                }
                for key, (sym, ast_decoder) in frames.items():
                    ok = verify_roundtrip(text, sym, lambda s, d=ast_decoder: expand_frame(d(s)))
                    cands.append({
                        "key": key, "payload": sym, "roundtrip_verified": ok,
                        "roundtrip_exact": ok and expand_frame(ast_decoder(sym)) == text,
                    })

        for key, enc, dec in (
            ("base85", UniversalExactCodec.encode_to_base85, UniversalExactCodec.decode_from_base85),
            ("runes", UniversalExactCodec.encode_to_runes, UniversalExactCodec.decode_from_runes),
        ):
            try:
                payload = enc(text)
            except Exception:
                continue
            ok = verify_roundtrip(text, payload, dec, normalize=None)  # lossless codecs must be byte-exact
            cands.append({"key": key, "payload": payload, "roundtrip_verified": ok, "roundtrip_exact": ok})

        for c in cands:
            c["tokens"] = count(c["payload"])
            c["chars"] = len(c["payload"])
            c["llm_readable"] = _LLM_READABLE.get(c["key"], False)
            c["mode_name"] = _MODE_NAMES[c["key"]]
        return cands

    # -------------------------------------------------------------
    # 1. INPUT STAGE
    # -------------------------------------------------------------
    def prepare_llm_input(
        self,
        raw_input_text: str,
        domain_mode: str = "auto",
        llm_readable_only: bool = True,
    ) -> Dict[str, Any]:
        """Choose the cheapest verified representation of `raw_input_text` and build the LLM prompt.

        domain_mode: 'auto' (default) and 'telemetry' (legacy name) both try the domain frame, which applies ONLY if
            the input is exactly the canonical sentence of the one pre-agreed grammar (whitespace-normalised); there is
            no keyword trigger, and 'telemetry' no longer forces the frame on other text. 'direct' disables
            compression.
        llm_readable_only: if True (default), only candidates an LLM can read given the prompt's protocol line are
            eligible (raw text and the Omega radix frame). The 3-char / 1-glyph frame codes and the zlib forms are still
            measured and listed in `candidates`; set False to let the router pick them for storage/transport.

        The chosen candidate always has at most as many tokens as the raw text, so the saving is >= 0. Limits: tokens
        are counted by `self.count_tokens` (tokenizer-specific; re-measure for another model), and savings are on the
        payload, not the whole prompt (see `prompt_saving_pct`).
        """
        count = self.count_tokens
        raw_tokens = count(raw_input_text)
        forced_direct = domain_mode == "direct"
        cands = self._candidates(raw_input_text, use_frame=not forced_direct)
        raw = cands[0]

        note = ""
        if domain_mode == "telemetry" and not any(c["key"].startswith("frame_") for c in cands):
            note = "domain_mode='telemetry' requested but the input is not the canonical frame sentence; no frame applied."

        def eligible(c: Dict[str, Any]) -> bool:
            if c["key"] == "direct":
                return True
            if forced_direct or not c["roundtrip_verified"]:
                return False
            return c["llm_readable"] or not llm_readable_only

        best = raw
        for c in cands:
            if not eligible(c):
                continue
            better = c["tokens"] < best["tokens"] or (
                c["tokens"] == best["tokens"] and _PREFERENCE.index(c["key"]) < _PREFERENCE.index(best["key"])
            )
            if c is not raw and better:
                best = c

        key = best["key"]
        mode_used = _MODE_NAMES["direct_forced"] if (forced_direct and key == "direct") else _MODE_NAMES[key]
        comp_tokens = best["tokens"]
        saving = _saving_pct(comp_tokens, raw_tokens)
        assert saving >= 0.0 and comp_tokens <= raw_tokens  # the router invariant; raw is always a candidate

        if key == "direct":
            llm_prompt = raw_input_text
        elif key == "frame_radix":
            llm_prompt = (
                f"[SYMLANG FRAME v1: {_FRAME_KEY} is one fixed 294,912-state template; slots are abbreviations]\n"
                f"INPUT_PAYLOAD: {best['payload']}\n"
                f"TASK: Process the state and, if you answer in the frame, emit Ω[...]; otherwise answer in plain text."
            )
        else:
            llm_prompt = (
                f"[SYMLANG CODEBOOK PAYLOAD ({key}); decodable only with the shared codebook, not by reading it]\n"
                f"INPUT_PAYLOAD: {best['payload']}"
            )
        prompt_tokens = count(llm_prompt)

        return {
            "original_input": raw_input_text,
            "original_tokens": raw_tokens,
            "compressed_symbol": best["payload"],
            "compressed_tokens": comp_tokens,
            "input_token_saving_pct": saving,
            "mode_used": mode_used,
            "mode": key,
            "roundtrip_verified": best["roundtrip_verified"],
            "roundtrip_exact": best["roundtrip_exact"],
            "token_counter": self.token_counter_name,
            "candidates": [
                {k: c[k] for k in ("key", "mode_name", "tokens", "chars", "llm_readable", "roundtrip_verified")}
                for c in cands
            ],
            "llm_prompt": llm_prompt,
            "llm_prompt_tokens": prompt_tokens,
            "prompt_saving_pct": _saving_pct(prompt_tokens, raw_tokens),
            "note": note,
        }

    # -------------------------------------------------------------
    # 2. OUTPUT STAGE
    # -------------------------------------------------------------
    def parse_llm_output(
        self, llm_raw_symbolic_output: str, codebook: bool = False, decode: bool = True
    ) -> Dict[str, Any]:
        """Decode an LLM reply: Omega radix blocks, zlib runes/base85, or (with `codebook=True`) 1-glyph / 3-char codes.

        `decode=False` returns the reply as plain text (mode 'Direct'); `execute_turn` uses it when the prompt was sent
        raw, because then nothing in the reply is a payload and text that merely looks like a symbol must stay text.

        Detection order: Omega radix blocks (expanded in place through the AST), then plane-1 runes (any code point
        >= U+10000, i.e. what `UniversalExactCodec.encode_to_runes` emits), then '§'-prefixed base85, otherwise
        'Direct'. Every decode attempt is
        wrapped: if it fails (not a valid DEFLATE stream, bad radix, value outside the frame) the reply is returned
        unchanged with mode 'Direct'. A lone plane-1 character or a 3-character string is only interpreted as a frame
        code when `codebook=True`, because otherwise an emoji or the word 'yes' would be decoded as a state.
        """
        count = self.count_tokens
        raw_symbol = llm_raw_symbolic_output.strip()
        comp_tokens = count(raw_symbol)
        human_text = raw_symbol
        mode_used = "Direct"

        if not decode:
            pass
        elif "Ω[" in raw_symbol:
            changed = False

            def _expand(m: "re.Match[str]") -> str:
                nonlocal changed
                try:
                    ast = MetaTierCodec.decode_layer8_radix(m.group(0))
                    text = self.decoder.decode_deterministic(ast.emit_tier6())
                except Exception:
                    return m.group(0)
                if text != ast.emit_tier6():
                    changed = True
                    return text
                return m.group(0)

            expanded = _RADIX_RE.sub(_expand, raw_symbol)
            if changed:
                human_text, mode_used = expanded, "SymRadix -> AST -> Human English"
        elif codebook and len(raw_symbol) == 1 and ord(raw_symbol) >= 0x10000:
            try:
                human_text = expand_frame(MetaTierCodec.decode_layer12_singular(raw_symbol))
                mode_used = "Frame/Singular -> AST -> Human English"
            except Exception:
                pass
        elif codebook and len(raw_symbol) == 3 and raw_symbol.isascii():
            try:
                human_text = expand_frame(MetaTierCodec.decode_layer11_triad(raw_symbol))
                mode_used = "Frame/Triad -> AST -> Human English"
            except Exception:
                pass
        elif any(ord(c) >= 0x10000 for c in raw_symbol):
            try:
                human_text = UniversalExactCodec.decode_from_runes(raw_symbol)
                mode_used = "UniversalRunes -> Lossless Text"
            except Exception:
                human_text, mode_used = raw_symbol, "Direct"
        elif raw_symbol.startswith("§"):
            try:
                human_text = UniversalExactCodec.decode_from_base85(raw_symbol)
                mode_used = "UniversalBase85 -> Lossless Text"
            except Exception:
                human_text, mode_used = raw_symbol, "Direct"

        human_tokens = count(human_text)
        return {
            "raw_symbol_from_llm": raw_symbol,
            "compressed_tokens": comp_tokens,
            "human_readable_output": human_text,
            "human_tokens": human_tokens,
            # Signed on purpose: a reply that costs more tokens than the text it expands to shows as negative.
            "output_token_saving_pct": _saving_pct(comp_tokens, human_tokens),
            "mode_used": mode_used,
        }

    # -------------------------------------------------------------
    # 3. END-TO-END TURN
    # -------------------------------------------------------------
    def execute_turn(
        self,
        user_input: str,
        llm_caller: Optional[Callable[[str], str]] = None,
        llm_readable_only: bool = True,
    ) -> Dict[str, Any]:
        """Input -> router -> LLM -> post-processor.

        Without `llm_caller` the LLM is SIMULATED as an identity echo of the compressed payload and the result has
        `simulated_llm: True`; the token figures then describe the encoding only, not any model's behaviour. With an
        `llm_caller`, its reply is parsed as-is (unchanged behaviour).
        """
        in_res = self.prepare_llm_input(user_input, llm_readable_only=llm_readable_only)

        simulated = llm_caller is None
        if simulated:
            raw_llm_reply = in_res["compressed_symbol"]  # explicit identity echo, not a model answer
        else:
            raw_llm_reply = llm_caller(in_res["llm_prompt"])

        codebook = in_res["mode"] in ("frame_triad", "frame_singular")
        out_res = self.parse_llm_output(raw_llm_reply, codebook=codebook, decode=in_res["mode"] != "direct")

        total_original = in_res["original_tokens"] + out_res["human_tokens"]
        total_compressed = in_res["compressed_tokens"] + out_res["compressed_tokens"]
        total_with_scaffold = in_res["llm_prompt_tokens"] + out_res["compressed_tokens"]

        return {
            "simulated_llm": simulated,
            "input_stage": in_res,
            "llm_intermediate_reply": raw_llm_reply,
            "output_stage": out_res,
            "total_token_metrics": {
                "total_standard_tokens": total_original,
                "total_symbolic_tokens": total_compressed,
                "overall_token_saving_pct": _saving_pct(total_compressed, total_original),
                "total_symbolic_tokens_with_prompt_scaffold": total_with_scaffold,
                "overall_saving_pct_with_prompt_scaffold": _saving_pct(total_with_scaffold, total_original),
                "token_counter": self.token_counter_name,
            },
        }
