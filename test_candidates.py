import tiktoken
import zlib
import base64

enc_cl100k = tiktoken.get_encoding("cl100k_base")
enc_o200k = tiktoken.get_encoding("o200k_base")

original_text = (
    "While the model starts, your PC can be slow or stop responding for 1-3 minutes "
    "(longest the first time). Strata loads 35-55 GB into your RAM and locks part of "
    "it for the graphics card. This is normal. Wait, and don't close the window. "
    "The window shows what Strata is doing."
)

candidates = {
    "Original English": original_text,
    
    "Tier 1: SymTelegraph (Dense Semantic Shorthand)": (
        "Model starting: PC slow/unresponsive 1-3m (longest 1st time). "
        "Strata loads 35-55GB into RAM, locks part for GPU. Normal. "
        "Wait; keep window open to view progress."
    ),
    
    "Tier 2: SymLogic-ASCII (First-Order Operator Logic)": (
        "^Model => PC(slow|freeze) @1-3m (*1st=max). "
        "Strata:35-55GB->RAM+GPU_lock. [NORMAL]. "
        "WAIT; !close(WIN): WIN=status(Strata)."
    ),
    
    "Tier 3: SymLogic-Unicode (Dense Mathematical Shorthand)": (
        "▶M⇒PC(∿|⏸) 1-3m (1st=max). "
        "Strata: 35-55GB→RAM⊞GPU. ✓Norm. "
        "⏳¬✕Win: Win=👁Strata."
    ),
    
    "Tier 4: SymAST (Context-Free Semantic Frame)": (
        "[INIT:M]>[PC:~0|⏸,DUR:1-3m,MAX:1st];"
        "[ALLOC:S,35-55GB,RAM+GPU];"
        "[NORM];[!EXIT(W),WAIT];[W=LOG:S]"
    ),
    
    "Tier 5: UltraGlyph (Conceptual Ideogram Shorthand)": (
        "🤖▶⇒💻🐢|🛑⏱1-3m(①=max). ⚡35-55G💾+🎮. ✔🆗. ⏳¬❌🪟(🪟=👁⚡)"
    ),
    
    "Tier 6: SymNano (Ultra-Minimalist Radix Syntax)": (
        "M.init>PC.lag(1-3m,1st=max)|S.ram(35-55G+gpu)|OK|wait;!x(win):win=obs(S)"
    ),
}

print(f"{'Tier / Representation':<55} | {'Chars':>5} | {'Bytes':>5} | {'cl100k':>6} | {'o200k':>6} | {'Char %':>6} | {'Tok %':>6}")
print("-" * 98)

orig_chars = len(original_text)
orig_toks = len(enc_cl100k.encode(original_text))

for name, text in candidates.items():
    chars = len(text)
    bytes_len = len(text.encode("utf-8"))
    tok_cl = len(enc_cl100k.encode(text))
    tok_o2 = len(enc_o200k.encode(text))
    char_comp = (1 - chars / orig_chars) * 100
    tok_comp = (1 - tok_cl / orig_toks) * 100
    print(f"{name:<55} | {chars:>5} | {bytes_len:>5} | {tok_cl:>6} | {tok_o2:>6} | {char_comp:>5.1f}% | {tok_comp:>5.1f}%")
