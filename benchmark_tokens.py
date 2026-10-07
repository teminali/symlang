"""Token-honest benchmark for SymLang's generic (lossless) codecs.

Characters are not what an LLM bills or reads: tokens are. This script measures what each representation of the SAME
content costs in real tokenizer tokens, so a "saving" is only claimed where tokens actually go down.

    python benchmark_tokens.py [file ...]        # default: a few text/code samples from this repo
    python benchmark_tokens.py --encodings cl100k_base o200k_base

For every sample it prints characters, UTF-8 bytes and tokens for: the raw text, zlib+Base85 (`encode_to_base85`) and zlib+runes
(`encode_to_runes`), plus the saving in each unit (negative = it got bigger). It also checks the lossless round trip.
Tokenizers other than OpenAI's (for example Qwen on a local Ollama) differ: measure with the target model's tokenizer before
trusting any of these figures for it (see `ollama_prompt_tokens` below for a cheap way).
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import tiktoken

from symlang.universal_codec import UniversalExactCodec as U

DEFAULT_FILES = ["README.md", "symlang/meta_tier.py", "symlang/pipeline.py", "index.html", "tests/test_pipeline.py"]


def ollama_prompt_tokens(text, model, host="http://127.0.0.1:11434"):
    """Prompt token count as the local model's own tokenizer sees it (one tiny generate call, no output)."""
    body = json.dumps({"model": model, "prompt": text, "stream": False, "options": {"num_predict": 1}}).encode()
    req = urllib.request.Request(host + "/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r).get("prompt_eval_count")


def pct(new, old):
    return 100.0 * (1.0 - new / old) if old else 0.0


def measure(text, encs, ollama_model=None):
    forms = {
        "raw": text,
        "base85": U.encode_to_base85(text),
        "runes": U.encode_to_runes(text),
    }
    assert U.decode_from_base85(forms["base85"]) == text, "base85 round trip failed"
    assert U.decode_from_runes(forms["runes"]) == text, "runes round trip failed"
    rows = []
    for name, s in forms.items():
        row = {"form": name, "chars": len(s), "bytes": len(s.encode("utf-8"))}
        for e in encs:
            row[e] = len(e_cache[e].encode(s, disallowed_special=()))
        if ollama_model:
            row["ollama"] = ollama_prompt_tokens(s, ollama_model)
        rows.append(row)
    return rows


e_cache = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--encodings", nargs="+", default=["cl100k_base"])
    ap.add_argument("--ollama-model", default=None, help="also count tokens with this local Ollama model's tokenizer")
    ap.add_argument("--max-chars", type=int, default=6000)
    a = ap.parse_args()
    for e in a.encodings:
        e_cache[e] = tiktoken.get_encoding(e)
    root = Path(__file__).parent
    files = a.files or DEFAULT_FILES
    cols = a.encodings + (["ollama"] if a.ollama_model else [])
    worst = 0
    for f in files:
        p = (root / f) if not Path(f).is_absolute() else Path(f)
        if not p.exists():
            print("skip (missing):", f)
            continue
        text = p.read_text(encoding="utf-8")[: a.max_chars]
        rows = measure(text, a.encodings, a.ollama_model)
        raw = rows[0]
        print(f"\n{f}  ({raw['chars']} chars, {raw['bytes']} bytes)")
        print(f"  {'form':8} {'chars':>7} {'bytes':>7} " + " ".join(f"{c[:12]:>13}" for c in cols))
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c)
                cells.append(f"{v if v is not None else '-':>6} ({pct(v, raw[c]):+5.0f}%)" if v is not None and raw.get(c) else f"{'-':>13}")
            print(f"  {r['form']:8} {r['chars']:>7} {r['bytes']:>7} " + " ".join(cells) +
                  f"   chars {pct(r['chars'], raw['chars']):+.0f}%  bytes {pct(r['bytes'], raw['bytes']):+.0f}%")
        worst = max(worst, rows[2].get(cols[0], 0) / max(1, raw.get(cols[0], 1)))
    print(f"\nWorst case: runes cost {worst:.1f}x the tokens of the raw text in {cols[0]}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
