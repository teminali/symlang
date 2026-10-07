# SymLang ⚡

```text
  ____                  _                             
 / ___| _   _ _ __ ___ | |    __ _ _ __   __ _  ___  
 \___ \| | | | '_ ` _ \| |   / _` | '_ \ / _` |/ _ \ 
  ___) | |_| | | | | | | |__| (_| | | | | (_| | (_) |
 |____/ \__, |_| |_| |_|_____\__,_|_| |_|\__, |\___/ 
        |___/                            |___/  ⚡
```

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Donate: Crypto](https://img.shields.io/badge/Donate-Crypto-yellow.svg)](DONATE.md)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange.svg)](#-status-and-limits)

> **Experimental symbolic encodings for LLM text, with a token-aware router that refuses to make things worse.**
> Two real results, scoped honestly below: (1) one pre-agreed 294,912-sentence grammar that bijects to a single symbol,
> and (2) a lossless generic codec that saves *characters* but not *tokens*. Everything else is a demo or a plan.
> Numbers here come from `benchmark.py`, `benchmark_tokens.py` and `pytest`, not from claims.

---

## ⚡ Quickstart

### 1. Installation
```bash
git clone https://github.com/teminali/symlang.git
cd symlang
pip install -e ".[dev]"
```

### 2. Python API
```python
from symlang.pipeline import NeuroSymbolicPipeline

pipeline = NeuroSymbolicPipeline()          # tokens counted with cl100k_base; pass token_counter=... for another tokenizer

# Pre-processor: builds every candidate form, counts tokens, picks the cheapest one that round-trips (never worse than raw)
r = pipeline.prepare_llm_input("any text or code")
print(r["mode_used"], r["original_tokens"], "->", r["compressed_tokens"], f"({r['input_token_saving_pct']:.1f}% saved)")

# Full turn. With no llm_caller the "LLM" is an explicit identity echo and the result says so: simulated_llm == True
res = pipeline.execute_turn("any text or code")
print(res["simulated_llm"], res["output_stage"]["human_readable_output"])
```

### 3. Command line
```bash
symlang encode "Your prompt text here" --tier ultra   # lossy abbreviation tiers (no decoder guarantee)
symlang meta-decode "Ω[M·PC·~0·1-3m·1st:max·S·35-55G·RAM·GPU·ok·win]" --layer radix   # back to Tier 6
symlang meta-decode '!!!' --layer triad               # Layer 11 (also --layer singular for Layer 12); strict: only canonical codes
symlang proof                                         # run the bijection proof for that grammar
python benchmark.py                                   # every tier on the canonical sentence, tokens counted, decoders actually run
python benchmark_tokens.py --encodings cl100k_base o200k_base   # token cost of the generic codecs
```

---

## 🔄 What the middleware does

```
[ text ] -> candidates: raw | frame (only if text IS the canonical sentence) | zlib+base85 | zlib+runes
         -> count tokens (cl100k_base by default, injectable)
         -> drop any candidate whose expansion does not reproduce the input (round-trip guard)
         -> pick the fewest tokens; ties and "nothing is cheaper" go to raw ("Direct (no saving found)", 0%)
```

The raw text is always a candidate, so a saving is never negative. The zlib forms are lossless but opaque: an LLM cannot
inflate DEFLATE, so by default (`llm_readable_only=True`) they are measured and listed in `result["candidates"]` but not
chosen for a prompt. Savings are on the payload; `llm_prompt_tokens` / `prompt_saving_pct` include the protocol line,
which for one short message can erase the saving (see below).

---

## 📊 Result 1: one pre-agreed grammar (a template frame, not text compression)

*Benchmark: the one 273-character sentence the grammar is built around. Tokens counted with tiktoken.*

| Representation | Chars | UTF-8 bytes | cl100k tokens (change) | o200k tokens (change) |
| :--- | :---: | :---: | :---: | :---: |
| Natural English (the canonical sentence) | 273 | 273 | 69 | 68 |
| Tier 1 SymTelegraph (lossy, no decoder guarantee) | 161 | 161 | 47 (-32%) | 47 (-31%) |
| Tier 3 SymNano (lossy, no decoder guarantee) | 72 | 72 | 39 (-43%) | 40 (-41%) |
| Tier 6 SymOperator | 56 | 72 | 48 (-30%) | 43 (-37%) |
| Layer 7 SymTensor | 49 | 61 | 41 (-41%) | 40 (-41%) |
| Layer 8 SymRadix `Ω[M·PC·~0·...]` | 47 | 58 | 33 (-52%) | 32 (-53%) |
| Layer 9B SymRune (6 CJK characters) | 6 | 18 | 7 (-90%) | 7 (-90%) |
| Layer 11 SymTriad `!!!` | 3 | 3 | 1 (-99%) | 1 (-99%) |
| Layer 12 SymSingular (1 code point) | 1 | 4 | 4 (-94%) | 4 (-94%) |

**How to read this.** The grammar has 11 slots with small closed domains: 4x3x4x4x4x2x4x2x2x4x3 = **294,912** sentences
(18.17 bits). Sender and receiver both hold the same codebook (`FRAME_PHRASES` + template in `symlang/decoder.py`), so one
state fits in one code point. **That is possible only because both sides share the codebook; it is not general text
compression.** A sentence outside the grammar cannot be encoded at all, and the token count of a 1-character glyph is
tokenizer-dependent (4 tokens here, not 1). A model that does not hold the codebook cannot read `!!!`; sending the
codebook costs more tokens than it saves on a single message.

What is verified: sentence -> AST -> sentence over all 294,912 states (every state expands to a distinct sentence and parses
back to itself), and AST <-> Layer 8 / 11 / 12 symbol over all 294,912 states. The test suite runs a sample of both; the
exhaustive run (`python proof.py`) takes 10-30 seconds; `--quick` checks a sample.

---

## 💻 Result 2: the generic lossless codec saves characters, costs tokens

```python
from symlang.universal_codec import UniversalExactCodec

code = """def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)"""

runes = UniversalExactCodec.encode_to_runes(code)         # zlib, two bytes per plane-1 character
assert UniversalExactCodec.decode_from_runes(runes) == code   # byte-for-byte lossless (tested)
```

It is exact, and it removes 40-85% of the *characters*. But LLMs read tokens, and DEFLATE bytes shown as characters tokenize
badly. First 6,000 characters of each file, `python benchmark_tokens.py symlang/meta_tier.py index.html test_universal.py --encodings cl100k_base o200k_base`:

| Sample | Raw tokens (cl100k / o200k) | zlib+Base85 tokens | zlib+runes tokens | Runes chars saved |
| :--- | :---: | :---: | :---: | :---: |
| `symlang/meta_tier.py` | 1670 / 1663 | 2302 / 2233 (1.4x / 1.3x) | 4555 / 4524 (2.7x / 2.7x) | 80% |
| `index.html` | 1900 / 1903 | 1704 / 1663 (0.9x / 0.9x) | 3437 / 3401 (1.8x / 1.8x) | 85% |
| `test_universal.py` | 921 / 922 | 1512 / 1478 (1.6x / 1.6x) | 3018 / 3003 (3.3x / 3.3x) | 77% |

**Runes cost 1.8-3.3x the tokens of the raw text; Base85 costs 0.9-1.6x** (it only wins on the dense markup file, and its
output is not readable by a model). Use this codec for storage and transport between your own systems, not as prompt
compression. Other tokenizers (for example Qwen on a local Ollama) differ: re-measure with the target model's tokenizer
(`--ollama-model`).

### What the router does with real inputs

`prepare_llm_input`, cl100k_base, default settings (`llm_readable_only=True`):

| Input | Raw tokens | Mode chosen | Tokens | Saving |
| :--- | :---: | :--- | :---: | :---: |
| the canonical sentence | 69 | Frame/Radix (canonical sentence, round trip verified) | 33 | 52.2% |
| `it is fine; you can close the window` | 9 | Direct (no saving found) | 9 | 0.0% |
| `fib()` Python, 4 lines | 30 | Direct (no saving found) | 30 | 0.0% |
| a JSON object | 50 | Direct (no saving found) | 50 | 0.0% |
| one English sentence | 30 | Direct (no saving found) | 30 | 0.0% |

The 52.2% is on the payload for the one sentence the frame exactly reproduces. With the protocol line that tells a model what
`Ω[...]` means, the whole one-shot prompt is 114 tokens against 69 raw; the frame pays off only when that line is amortised
across many messages (a system prompt) or when both ends already hold the codebook. Any near-miss of the canonical sentence,
including one that reverses its meaning, goes Direct.

---

## 📐 The bijection, scoped

Every sentence of the grammar is an 11-slot AST state, so `|Ω| = 294,912` and `H(Ω) = log2(294,912) = 18.17 bits`. Unicode has
1,114,112 code points, so each state can map to one code point.

* Parser `P` (sentence -> AST) and synthesizer `E` (AST -> sentence) satisfy `E(P(s)) = s` for every sentence `s` of the grammar
  (checked exhaustively, see above), and `P(E(a)) = a` for every state `a`.
* This says nothing about text outside the grammar. Free-form input is not parsed, not guessed and not "decoded": the pipeline
  sends it Direct.
* Out-of-domain slot values and state ids >= 294,912 raise `ValueError` (they used to be remapped silently).
* "One state, one code" is enforced on the decode side too: `decode_layer11_triad` accepts only the canonical 3 characters
  (each in `'!'..'u'`, value < 294,912) and raises `ValueError` for any other spelling such as the alias `'!v!'`. The CLI and
  the library behave the same; a decoder that accepted several spellings of one state would not be a bijection.

```bash
python3 proof.py    # runs the proof demo for the grammar
```

---

## 🖥️ Interactive Web Studio

```bash
open index.html
```
A dark-mode viewer of the tiers and the AST reconstruction for the grammar above.

---

## 📈 Results so far

> The corpus behind these numbers (`corpus/data/`, `symlang/library/data/`) is derived from a private repository and is **not published**. The numbers below are the measured aggregates; to reproduce them, point `DETERMINISTIC_CODER_REPO` at your own checkout and run `python -m corpus.build_corpus` (the data-dependent tests skip when it is absent).

Both are measured on recorded data of the Deterministic Coder app (the "Kernel" spec language), at small n. Read the full
write-ups before quoting a number.

* **Wire formats: [docs/RESULTS_formats.md](docs/RESULTS_formats.md).** Dense formats do **not** beat minified JSON. On the
  15 parseable 14B replies, short keys save +0.1%, positional arrays -0.7% (cl100k) to +1.6% (o200k) and S-expressions
  -3.5% to -0.2%: within noise of what the model already writes. The one-rule-per-line `line` format saves about **7% of tokens on real 14B replies** (7.3% cl100k, 10.0%
  o200k, n=15), but Ollama's `format=` takes JSON Schema only, so `line` cannot be grammar-constrained there (it needs a
  GBNF runtime). Pretty-to-minified JSON is the big gap and it is already taken. Qwen's own tokenizer was not measured.
* **Library learning: [docs/RESULTS_library.md](docs/RESULTS_library.md).** Learning reusable blocks by minimum description
  length works as a mechanism (exact round trip, a hand-written helper rediscovered). In-sample it saves **24% of nodes**
  net of the library, but only **4.6% of tokens held out** (leave-one-out, gross, library not charged), and the in-sample token
  net is **-2.3%**. n=6 specs: an existence check, not a rate. Scale the corpus (30+ specs) before claiming anything.

---

## 🧭 Status and limits

**Real**
* The AST <-> symbol bijection for the one grammar (Tier 6, Layers 7, 8, 11, 12), including the sentence frame, verified
  over all 294,912 states.
* The lossless zlib round trip (`UniversalExactCodec`), tested on code, indentation, newlines, JSON and unicode.
* `LosslessSymbolicCodec` (14 fixed phrases to one byte, ASCII literals, 4 bytes for any other code point) round-trips every
  `str` exactly, non-ASCII, emoji and lone surrogates included (seeded property test in `tests/test_lossless.py`). It is not
  a compressor for arbitrary text: non-ASCII characters get bigger, and its glyph string costs more tokens than the raw text.
* The router: token-counted candidates, never worse than raw, round-trip guarded, `simulated_llm` flag on stub turns.

**Demo or stub**
* `execute_turn` without an `llm_caller` is an identity echo, not a model. Nothing in this repo has measured a real model's
  behaviour, latency or output quality with these encodings; there is no speed-up claim.
* `run_pipeline_demo.py` uses a hardcoded reply. The lossy tiers (Tier 1-4, ultra-glyph) have no decoder guarantee.
* `verify_semantic_fidelity` is a keyword smoke test, not a meaning check.

**Not claimed**
* General prompt compression, token savings on arbitrary text, "zero information loss" outside the grammar, or speed.

**Planned: SymLang 2**
* Two spikes have results (see "Results so far"); neither is wired into the router yet. Work lives in `corpus/`,
  `benchmark/` and `symlang/library/`. Next: a bigger corpus before any claim, then a frame engine with verified round
  trips and the token-aware router extended to use them.

---

## 📂 Project Structure

```text
symlang/
├── symlang/
│   ├── pipeline.py           # Token-aware router, round-trip guard, turn runner
│   ├── meta_tier.py          # AST parser & Layers 7-12 for the one grammar
│   ├── decoder.py            # AST -> sentence expansion, sentence -> AST, prompts
│   ├── universal_codec.py    # Lossless zlib codec (runes / Base85)
│   ├── lossless.py           # Exact phrase-codebook codec (any str round-trips)
│   ├── library/              # Library-learning (MDL) spike
│   ├── encoder.py            # Lossy abbreviation tiers
│   ├── llm_bridge.py         # Prompt/parse adapter for the grammar
│   └── metrics.py            # Token counters & Shannon entropy
├── tests/                    # pytest: pipeline, meta-tier, lossless codec, CLI, formats, library, honesty tests
├── benchmark.py              # Every tier on the canonical sentence: tokens counted, decoders run
├── benchmark_tokens.py       # Token cost of the generic codecs
├── benchmark/                # Wire-format and token-metric tooling (docs/RESULTS_formats.md)
├── corpus/                   # Recorded Kernel specs and replies the SymLang 2 spikes measure
├── docs/                     # RESULTS_formats.md, RESULTS_library.md
├── proof.py                  # Bijection proof demo for the grammar
├── cli.py                    # CLI
├── index.html                # Interactive viewer
└── pyproject.toml
```

Run the tests with `python -m pytest -q`.

---

## 💖 Support this work

`symlang` is free and open-source. If it saved you time, [a coffee's worth of crypto](DONATE.md) is a genuinely great way to say
so. It stays free either way.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
