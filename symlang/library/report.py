"""Run the library-learning spike on the six Deterministic Coder preset specs and print the report.

    ./.venv/bin/python -m symlang.library              # nodes objective
    ./.venv/bin/python -m symlang.library --objective tokens
    ./.venv/bin/python -m symlang.library --refresh    # re-dump presets.json from the other repo
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .learner import (Cost, Result, compress, def_text, expand, leave_one_out, ntok, roundtrip_ok,
                      token_report, transfer)
from .tree import Node, from_json, iter_sites

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "presets.json")
def _find_dc_repo():
    """Private Deterministic Coder checkout (needed only to REFRESH the data): $DETERMINISTIC_CODER_REPO, else a sibling directory."""
    cands = [os.environ.get("DETERMINISTIC_CODER_REPO", "")]
    cands += [os.path.join(HERE, *([".."] * n), "deterministic-coder") for n in (3, 4)]
    for c in cands:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return ""


KERNEL_PRESETS_JS = os.path.join(_find_dc_repo() or "deterministic-coder", "ui", "public", "backend", "kernel_presets.js")
FORM_LIST = ["counter", "todo", "bmi-calculator"]


def refresh(src: str = KERNEL_PRESETS_JS) -> None:
    out = subprocess.run(["node", os.path.join(HERE, "dump_presets.js"), src], check=True,
                         capture_output=True, text=True).stdout
    json.loads(out)
    with open(DATA, "w") as f:
        f.write(out)


def load_presets() -> dict:
    with open(DATA) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- pseudo-code


def pseudo(n: Node, ind: int = 0, width: int = 96) -> str:
    """Human readable rendering: holes as ?i, calls as name(args)."""
    one = _inline(n)
    pad = "  " * ind
    if len(one) + len(pad) <= width or not n.kids:
        return one
    if n.kind == "D":
        rows = [f"{pad}  {k}: {pseudo(c, ind + 1, width)}," for k, c in zip(n.keys, n.kids)]
        return "{\n" + "\n".join(rows) + f"\n{pad}}}"
    if n.kind == "S":
        rows = [f"{pad}  {pseudo(c, ind + 1, width)}," for c in n.kids]
        return "[\n" + "\n".join(rows) + f"\n{pad}]"
    rows = [f"{pad}  {pseudo(c, ind + 1, width)}," for c in n.kids]
    return f"{n.val}(\n" + "\n".join(rows) + f"\n{pad})"


def _inline(n: Node) -> str:
    if n.kind == "L":
        return json.dumps(n.val, ensure_ascii=False)
    if n.kind == "H":
        return f"?{n.val}"
    if n.kind == "D":
        return "{" + ", ".join(f"{k}: {_inline(c)}" for k, c in zip(n.keys, n.kids)) + "}"
    if n.kind == "S":
        return "[" + ", ".join(_inline(c) for c in n.kids) + "]"
    return f"{n.val}(" + ", ".join(_inline(c) for c in n.kids) + ")"


def guess_title(p: Node) -> str:
    """Cosmetic only: a label from the most telling constants, so a human can scan the library."""
    if p.kind == "D":
        d = dict(zip(p.keys, p.kids))
        tag = d.get("tag")
        parts = []
        if tag is not None and tag.kind == "L":
            parts.append(f"<{tag.val}>")
        elif "tag" in d:
            parts.append("element")
        cls = d.get("class")
        if cls is not None and cls.kind == "L":
            parts.append(f".{cls.val}")
        if "set" in d:
            parts.append("state-set rule" + (" (guarded)" if "when" in d else ""))
        if "on" in d and "do" in d:
            parts.append("event -> actions")
        if "each" in d:
            parts.append("list loop")
        if "bind" in d:
            parts.append("bound input")
        if "persist" in d:
            parts.append("persisted state slot")
        if "aria-pressed" in d or "aria-label" in d:
            parts.append("aria")
        if parts:
            return " ".join(parts)
        return "record {" + ",".join(p.keys[:4]) + "}"
    if p.kind == "S":
        return f"list[{len(p.kids)}]"
    return "call"


# ----------------------------------------------------------------------------- the report


def pct(a: int, b: int) -> str:
    return f"{100.0 * (b - a) / b:5.1f}%" if b else "  n/a"


def run(objective: str = "nodes", top: int = 12) -> str:
    corpus = {k: from_json(v) for k, v in load_presets().items()}
    out: list[str] = []
    w = out.append
    cost = Cost(objective)

    res = compress(corpus, objective)
    assert roundtrip_ok(res)
    sz = res.sizes()
    tk = token_report(res)
    w(f"== library learned on all six presets (objective: {objective}) ==")
    w(f"abstractions: {len(res.library)}    round trip expand(compress(x)) == x: {roundtrip_ok(res)}")
    w(f"{'spec':16} {'orig nodes':>10} {'comp nodes':>10} {'saved':>7} | {'orig tok':>8} {'comp tok':>8} {'saved':>7}")
    for k in sorted(corpus):
        w(f"{k:16} {sz['orig'][k]:10d} {sz['comp'][k]:10d} {pct(sz['comp'][k], sz['orig'][k]):>7} | "
          f"{tk['orig'][k]:8d} {tk['comp'][k]:8d} {pct(tk['comp'][k], tk['orig'][k]):>7}")
    O, C = sum(sz["orig"].values()), sum(sz["comp"].values())
    TO, TC = sum(tk["orig"].values()), sum(tk["comp"].values())
    w(f"{'TOTAL (gross)':16} {O:10d} {C:10d} {pct(C, O):>7} | {TO:8d} {TC:8d} {pct(TC, TO):>7}")
    lib_nodes = res.library.size(Cost("nodes"))
    w(f"{'library':16} {lib_nodes:10d} nodes{'':16}| {tk['library']:8d} tokens")
    w(f"{'TOTAL (net)':16} {C + lib_nodes:10d} {pct(C + lib_nodes, O):>7}{'':13}| {TC + tk['library']:8d} {pct(TC + tk['library'], TO):>7}"
      "   <- MDL: corpus|library + library")

    w("")
    w(f"== leave-one-out (learn on 5, compress the held-out 6th), objective: {objective} ==")
    loo = leave_one_out(corpus, objective)
    w(f"{'held out':16} {'orig':>6} {'LOO saved':>10} {'LOO %':>7} {'in-sample saved':>16} {'in %':>7} {'used/learned':>13} | "
      f"{'LOO tok saved':>13} {'in tok saved':>12} (of orig tok)")
    tl = ti = to = ttl = tti = tto = 0
    for k, r in loo["rows"].items():
        w(f"{k:16} {r['orig']:6d} {r['loo_saving']:10d} {pct(r['loo_comp'], r['orig']):>7} "
          f"{r['in_saving']:16d} {pct(r['in_comp'], r['orig']):>7} {r['abstractions_used']:>6}/{r['abstractions_learned']:<6} | "
          f"{r['loo_tok_saving']:13d} {r['in_tok_saving']:12d} ({r['orig_tok']})")
        tl += r["loo_saving"]; ti += r["in_saving"]; to += r["orig"]
        ttl += r["loo_tok_saving"]; tti += r["in_tok_saving"]; tto += r["orig_tok"]
    w(f"{'TOTAL':16} {to:6d} {tl:10d} {100.0 * tl / to:6.1f}% {ti:16d} {100.0 * ti / to:6.1f}%   "
      f"(LOO keeps {100.0 * tl / ti:.0f}% of the in-sample saving)")
    w(f"{'TOTAL tokens':16} {tto:6d} {ttl:10d} {100.0 * ttl / tto:6.1f}% {tti:16d} {100.0 * tti / tto:6.1f}%   "
      f"(in real cl100k tokens; LOO keeps {100.0 * ttl / max(tti, 1):.0f}%)")

    w("")
    w(f"== transfer: learn on {FORM_LIST} only, apply to the others ({objective}) ==")
    tr = transfer(corpus, FORM_LIST, objective)
    w(f"abstractions learned from the 3 form/list presets: {len(tr['result'].library)}")
    for k, r in tr["rows"].items():
        w(f"{k:16} {'(train)' if r['train'] else '(held out)':11} orig {r['orig']:4d} -> {r['comp']:4d}  saved {r['saving']:4d} ({100.0 * r['saving'] / r['orig']:.1f}%)   tokens {r['orig_tok']} saved {r['tok_saving']}")

    w("")
    w(f"== top {top} abstractions by measured utility (objective: {objective}) ==")
    for st in sorted(res.steps, key=lambda s: -s.utility)[:top]:
        a = res.library.defs[st.name]
        params = ", ".join(f"?{i}" for i in range(a.arity))
        w(f"\n{st.name}({params})   # {guess_title(a.pattern)}   uses={st.uses}  utility={st.utility} ({objective})")
        w("  = " + pseudo(expand(a.body, res.library), 1))
    sample = None
    for k in sorted(res.rewritten):
        for _, n in iter_sites(res.rewritten[k]):
            if n.kind == "C" and n.val == res.steps[0].name:
                sample = (k, n)
                break
        if sample:
            break
    if sample:
        w(f"\nexample use in {sample[0]}: {pseudo(sample[1], 0, 200)}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m symlang.library")
    ap.add_argument("--objective", choices=["nodes", "tokens"], default="nodes")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--refresh", action="store_true", help="re-dump presets.json via node from the deterministic-coder repo")
    a = ap.parse_args(argv)
    if a.refresh:
        refresh()
    print(run(a.objective, a.top))


if __name__ == "__main__":
    sys.exit(main())
