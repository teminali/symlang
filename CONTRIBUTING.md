# Contributing to SymLang ⚡

Thanks for your interest in SymLang, an **alpha** project: symbolic encodings for LLM text, a lossless zlib codec, and a
token-aware, round-trip-verified router. Read "Status and limits" in the README first so you know what is real and what is a stub.

## 🛠 Development Setup

1. **Clone and Install:**
   ```bash
   git clone https://github.com/teminali/symlang.git
   cd symlang
   pip install -e ".[dev]"
   ```

2. **Run Tests:**
   ```bash
   python -m pytest -q
   python proof.py
   python benchmark_tokens.py --encodings cl100k_base o200k_base
   ```

3. **Guidelines:**
   - Adhere to PEP-8.
   - Every number you publish (README, PR text) must come from a command in this repo. Count **tokens** with the target
     tokenizer, not just characters; a smaller string can cost more tokens.
   - A lossy or frame-based path may only be used if expanding it reproduces the input. Use `symlang.pipeline.verify_roundtrip`
     and add a test that fails when the round trip breaks (see `tests/test_honest_pipeline.py`).
   - No keyword-triggered canned output. A decoder must derive text from the parsed AST, and return unknown input unchanged.
   - Codecs for the fixed grammar must keep `parse(expand(ast)) == ast` and `expand(parse(s)) == s` for every state, and raise
     on out-of-domain input instead of remapping it.
   - New symbolic tiers need tests and a token benchmark; say plainly if they cost more tokens than the raw text.

## 🤝 Pull Request Process
1. Fork the repo and branch from the default branch.
2. Make sure `python -m pytest -q` and `python proof.py` pass.
3. Describe the change with measured numbers (command + output), including any case where the change makes things worse.
