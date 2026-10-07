# Contributing to SymLang ⚡

Thank you for your interest in contributing to **SymLang**, the high-performance neuro-symbolic compression engine for LLMs!

## 🛠 Development Setup

1. **Clone and Install:**
   ```bash
   git clone https://github.com/your-username/symlang.git
   cd symlang
   pip install -e ".[dev]"
   ```

2. **Run Tests:**
   ```bash
   python3 -m unittest discover -s tests
   python3 proof.py
   ```

3. **Code Style & Guidelines:**
   - Adhere to PEP-8.
   - All deterministic codecs must preserve $E(P(s)) \equiv s$ (100% mathematical reversibility).
   - Any new symbolic tier must include test coverage and tokenization benchmark verification.

## 🤝 Pull Request Process
1. Fork the repo and create your branch from `main`.
2. Ensure all unit tests and the formal bijection proof (`proof.py`) pass.
3. Submit your PR with a clear description of changes and benchmark numbers!
