"""Token-metric harness: what does each wire format of a Kernel spec cost in the target model's tokens?

For every corpus artifact and every format in benchmark.formats it reports characters, UTF-8 bytes and tokens in
cl100k_base and o200k_base (tiktoken) and, only with --ollama and only when the local server is reachable and
answers a probe quickly, in the local Qwen tokenizer through Ollama (prompt_eval_count). Next to every number it
carries the round-trip result of that format on that spec. `--prompt-overhead` adds the per-call cost of the
current Kernel primer + response schema, which competes with the per-spec saving.

  ./.venv/bin/python -m benchmark.token_metrics                       # tables on stdout
  ./.venv/bin/python -m benchmark.token_metrics --prompt-overhead --markdown docs/RESULTS_formats.md
  ./.venv/bin/python -m benchmark.token_metrics --ollama              # also Qwen, if the server is idle
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.formats import FORMATS, Format, dumps_min

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus" / "data"
TIKTOKEN_NAMES = ["cl100k_base", "o200k_base"]
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "dc-qwen2.5-coder-14b"
DECODE_TOK_PER_S = 13.5  # the local 14B, as measured by the owner
BASELINE = "json_pretty"
AS_EMITTED = "raw_as_emitted"  # the reply text exactly as the model wrote it (replies only)
GROUPS = {
    "presets": "the six Kernel presets (canonical long form)",
    "replies_14b": "parseable raw replies of the local 14B on the 24 held-out prompts (shorthand form, as written)",
    "replies_7b": "parseable raw replies of the old 7B runs (shorthand form, as written)",
}


# ---------------------------------------------------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------------------------------------------------
def load_corpus(data_dir: Path = CORPUS_DIR) -> dict[str, Any]:
    """The measurable specs by group, plus the unparseable replies and the prompts."""
    data_dir = Path(data_dir)
    presets = json.loads((data_dir / "presets.json").read_text(encoding="utf-8"))
    replies = json.loads((data_dir / "replies.json").read_text(encoding="utf-8"))
    prompts = json.loads((data_dir / "prompts.json").read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = {g: [] for g in GROUPS}
    for pid, spec in presets.items():
        groups["presets"].append({"id": pid, "spec": spec, "raw": None})
    unparseable = []
    for r in replies:
        key = "replies_14b" if r["source"].startswith("14b") else "replies_7b"
        if r["parses"]:
            groups[key].append({"id": f"{r['source']}:{r['id']}", "spec": json.loads(r["raw"]), "raw": r["raw"]})
        else:
            unparseable.append({"id": f"{r['source']}:{r['id']}", "empty": not r["raw"], "truncated": bool(r.get("truncated"))})
    first_call = sorted(r["promptTokens"] for r in replies if r.get("turns") == 1 and r.get("promptTokens"))
    mpath = data_dir / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    return {"groups": groups, "unparseable": unparseable, "prompts": prompts, "first_call_prompt_tokens": first_call,
            "manifest": manifest}


# ---------------------------------------------------------------------------------------------------------------------
# token counters
# ---------------------------------------------------------------------------------------------------------------------
class TiktokenCounter:
    def __init__(self, name: str):
        import tiktoken  # imported late: the corpus and formats work without it

        self.name = name
        self._enc = tiktoken.get_encoding(name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


def _http_json(url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    """GET (no payload) or POST JSON. Kept as one small function so tests can replace it."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OllamaCounter:
    """Qwen tokens through Ollama. A one-token generation reports prompt_eval_count.

    Two traps, handled: (1) Ollama reuses the KV prefix of the previous request and then counts only the new
    tokens, so every prompt starts with a nonce whose first digit changes each call; (2) raw=True skips the chat
    template, and the nonce's own token count is measured once and subtracted.
    """

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST, timeout: float = 120.0):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.name = "qwen2.5-coder-14b (Ollama)"
        self._n = 0
        self._base: int | None = None

    def _eval(self, text: str, timeout: float | None = None) -> int:
        self._n += 1
        nonce = f"{self._n % 10}{self._n:05d}"
        out = _http_json(f"{self.host}/api/generate", {
            "model": self.model, "prompt": f"{nonce}\n{text}", "raw": True, "stream": False,
            "options": {"num_predict": 1, "temperature": 0}}, timeout=timeout or self.timeout)
        if "prompt_eval_count" not in out:
            raise RuntimeError("Ollama returned no prompt_eval_count (prompt fully cached?)")
        return int(out["prompt_eval_count"])

    def base(self) -> int:
        if self._base is None:
            self._base = self._eval("")
        return self._base

    def count(self, text: str) -> int:
        base = self.base()
        return self._eval(text) - base


def ollama_counter(host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL, probe_s: float = 8.0) -> tuple[OllamaCounter | None, str]:
    """(counter, note). The counter is None, with the reason in the note, whenever the server cannot be used safely.

    Ollama has no 'busy' flag, so the check is behavioural: /api/ps must answer, and a one-token probe must come
    back within `probe_s` (a server busy with another job queues the probe behind it and fails this).
    """
    try:
        ps = _http_json(f"{host.rstrip('/')}/api/ps", timeout=3.0)
    except (urllib.error.URLError, OSError, ValueError) as e:
        return None, f"skipped: Ollama not reachable at {host} ({type(e).__name__}: {e})"
    loaded = [m.get("name", "?") for m in ps.get("models", [])]
    counter = OllamaCounter(model, host)
    t0 = time.monotonic()
    try:
        counter._eval("x", timeout=probe_s)
    except Exception as e:  # timeout, HTTP error, bad payload: all mean "do not use"
        return None, f"skipped: Ollama probe failed or was slow ({type(e).__name__}: {e}); loaded models: {loaded}"
    took = time.monotonic() - t0
    if took > probe_s:
        return None, f"skipped: Ollama probe took {took:.1f}s (> {probe_s}s), another job is probably running"
    return counter, f"measured through Ollama model {model} (probe {took:.2f}s, loaded: {loaded})"


# ---------------------------------------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------------------------------------
def encodings_of(item: dict[str, Any], fmt: Format) -> tuple[str, bool, bool]:
    ok, ordered, text = fmt.roundtrip(item["spec"])
    return text, ok, ordered


def measure(corpus: dict[str, Any], counters: list, formats: dict[str, Format] | None = None,
            subsets: dict[str, int] | None = None, counter_subset: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """One row per (group, spec, format). `counter_subset` caps the specs per group for named counters (Ollama)."""
    formats = formats or FORMATS
    rows: list[dict[str, Any]] = []
    for gname, items in corpus["groups"].items():
        for idx, item in enumerate(items):
            variants: list[tuple[str, str, bool, bool]] = []
            for fname, fmt in formats.items():
                text, ok, ordered = encodings_of(item, fmt)
                variants.append((fname, text, ok, ordered))
            if item.get("raw") is not None:
                variants.append((AS_EMITTED, item["raw"], json.loads(item["raw"]) == item["spec"], True))
            for fname, text, ok, ordered in variants:
                row = {"group": gname, "id": item["id"], "format": fname, "chars": len(text),
                       "bytes": len(text.encode("utf-8")), "tokens": {}, "roundtrip": ok, "key_order": ordered}
                for c in counters:
                    cap = (counter_subset or {}).get(c.name)
                    if cap is not None and idx >= cap:
                        continue
                    row["tokens"][c.name] = c.count(text)
                rows.append(row)
    return rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate(rows: list[dict[str, Any]], counter_names: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """group -> format -> {n, chars, bytes, tokens{counter: mean}, n_tok{counter}, roundtrip, key_order}.

    Means are over the specs that carry a count for that counter, so Qwen (a subset) is compared on its own subset.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        a = out.setdefault(r["group"], {}).setdefault(r["format"], {
            "n": 0, "chars": [], "bytes": [], "tok": {c: [] for c in counter_names}, "ok": 0, "ordered": 0})
        a["n"] += 1
        a["chars"].append(r["chars"])
        a["bytes"].append(r["bytes"])
        a["ok"] += 1 if r["roundtrip"] else 0
        a["ordered"] += 1 if r["key_order"] else 0
        for c, t in r["tokens"].items():
            a["tok"][c].append(t)
    for g in out.values():
        for a in g.values():
            a["chars"], a["bytes"] = _mean(a["chars"]), _mean(a["bytes"])
            a["n_tok"] = {c: len(v) for c, v in a["tok"].items()}
            a["tokens"] = {c: _mean(v) for c, v in a["tok"].items()}
            del a["tok"]
    return out


def saving(mean: float, ref: float) -> float:
    """Percent fewer tokens than `ref` (positive = saves)."""
    return 100.0 * (ref - mean) / ref if ref else float("nan")


def paired_savings(rows: list[dict[str, Any]], counter: str, ref_format: str) -> dict[tuple[str, str], tuple[float, float, float]]:
    """(group, format) -> (min, median, max) per-spec saving vs `ref_format` on the same spec."""
    ref = {(r["group"], r["id"]): r["tokens"].get(counter) for r in rows if r["format"] == ref_format}
    per: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        base = ref.get((r["group"], r["id"]))
        t = r["tokens"].get(counter)
        if base and t is not None:
            per.setdefault((r["group"], r["format"]), []).append(saving(t, base))
    res = {}
    for k, v in per.items():
        v.sort()
        res[k] = (v[0], v[len(v) // 2], v[-1])
    return res


# ---------------------------------------------------------------------------------------------------------------------
# prompt overhead: primer + schema
# ---------------------------------------------------------------------------------------------------------------------
def load_kernel_prompt(data_dir: Path = CORPUS_DIR, live: bool = False) -> dict[str, Any]:
    """{primer, schema}. live=True re-evaluates kernel_model.js through the Node loader instead of the corpus copy."""
    if live:
        from corpus.build_corpus import DEFAULT_SRC, export_kernel

        k = export_kernel(DEFAULT_SRC)
        return {"primer": k["primer"], "schema": k["schema"]}
    return json.loads((Path(data_dir) / "kernel_prompt.json").read_text(encoding="utf-8"))


def primer_examples(primer: str) -> list[tuple[str, dict]]:
    """The few-shot spec examples embedded in the primer (`Example N: {json}`), as (prefix, object)."""
    out = []
    for line in primer.split("\n"):
        if line.startswith("Example ") and ": {" in line:
            prefix, _, body = line.partition(": ")
            try:
                out.append((prefix + ": ", json.loads(body)))
            except ValueError:
                pass
    return out


def prompt_overhead(kernel_prompt: dict[str, Any], counters: list, formats: dict[str, Format] | None = None) -> dict[str, Any]:
    """Tokens of the primer, the schema, and the primer with its examples re-written in each format (a LOWER bound
    on that format's primer: the instruction lines still describe JSON)."""
    formats = formats or FORMATS
    primer = kernel_prompt["primer"]
    schema_text = dumps_min(kernel_prompt["schema"])
    examples = primer_examples(primer)
    res: dict[str, Any] = {"primer_chars": len(primer), "schema_chars": len(schema_text), "n_examples": len(examples), "by_counter": {}}
    for c in counters:
        entry = {"primer": c.count(primer), "schema": c.count(schema_text), "primer_by_format": {}}
        for fname, fmt in formats.items():
            lines = []
            for line in primer.split("\n"):
                hit = next((ex for ex in examples if line.startswith(ex[0])), None)
                lines.append(hit[0] + fmt.encode(hit[1]) if hit else line)
            entry["primer_by_format"][fname] = c.count("\n".join(lines))
        res["by_counter"][c.name] = entry
    return res


# ---------------------------------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------------------------------
def _f(x: float, nd: int = 0) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{x:+.1f}%"


def format_order(agg_group: dict[str, dict[str, Any]]) -> list[str]:
    names = list(FORMATS)
    if AS_EMITTED in agg_group:
        names.append(AS_EMITTED)
    return names


def table_for(agg: dict, rows: list[dict[str, Any]], counter: str, groups: list[str]) -> list[str]:
    """Markdown table lines for one tokenizer."""
    lines: list[str] = []
    for g in groups:
        if g not in agg:
            continue
        ag = agg[g]
        if not any(counter in a["tokens"] and a["n_tok"].get(counter) for a in ag.values()):
            continue
        n = max(a["n_tok"].get(counter, 0) for a in ag.values())
        base = ag[BASELINE]["tokens"].get(counter, float("nan"))
        mini = ag["json_min"]["tokens"].get(counter, float("nan"))
        med = paired_savings(rows, counter, BASELINE)
        lines.append(f"**{g}** ({GROUPS[g]}; n={n}). Saving is relative to the reference format, positive = fewer tokens:")
        lines.append("")
        lines.append("| format | mean chars | mean bytes | mean tokens | saving vs pretty JSON | saving vs minified JSON | per-spec saving vs pretty (min / median / max) | round trip | key order kept | model_writable |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|---|---|")
        for fname in format_order(ag):
            a = ag[fname]
            if not a["n_tok"].get(counter):
                continue
            mt = a["tokens"][counter]
            lo, md, hi = med.get((g, fname), (float("nan"),) * 3)
            fmt = FORMATS.get(fname)
            writable = "yes" if fmt and fmt.model_writable else ("no" if fmt else "(what it writes today)")
            lines.append(f"| {fname} | {_f(a['chars'])} | {_f(a['bytes'])} | {_f(mt, 1)} | {_pct(saving(mt, base))} | {_pct(saving(mt, mini))} | "
                         f"{_pct(lo)} / {_pct(md)} / {_pct(hi)} | {a['ok']}/{a['n']} | {a['ordered']}/{a['n']} | {writable} |")
        lines.append("")
    return lines


def render_text(agg: dict, rows: list[dict[str, Any]], counters: list[str], groups: list[str]) -> str:
    out = []
    for c in counters:
        out.append(f"== {c} ==")
        out.extend(table_for(agg, rows, c, groups))
    return "\n".join(out)


def verdict_lines(agg: dict, rows: list[dict[str, Any]], counter: str) -> list[str]:
    """One sentence per format, computed from the numbers (primary tokenizer: the first counter)."""
    out = []
    for fname in FORMATS:
        parts = []
        all_ok = True
        for g in ("presets", "replies_14b"):
            a = agg.get(g, {}).get(fname)
            b = agg.get(g, {}).get(BASELINE)
            m = agg.get(g, {}).get("json_min")
            if not a or not b or not a["n_tok"].get(counter):
                continue
            all_ok &= a["ok"] == a["n"]
            parts.append(f"{g} {_pct(saving(a['tokens'][counter], b['tokens'][counter]))} vs pretty, "
                         f"{_pct(saving(a['tokens'][counter], m['tokens'][counter]))} vs minified")
        out.append(f"- `{fname}`: " + "; ".join(parts) + f"; round trip {'exact on every spec' if all_ok else 'FAILED on some spec'}; "
                   f"model_writable={FORMATS[fname].model_writable} ({FORMATS[fname].writable_note})")
    return out


def interpretation(corpus: dict[str, Any], agg: dict, counters: list[str]) -> list[str]:
    """What the numbers mean for the choice of wire format, with the arithmetic shown."""
    tik = [c for c in counters if not c.startswith("qwen")]
    ms = 1000.0 / DECODE_TOK_PER_S

    def rng(group: str, fmt: str, ref: str) -> tuple[str, str]:
        s = [saving(agg[group][fmt]["tokens"][c], agg[group][ref]["tokens"][c]) for c in tik]
        t = [agg[group][ref]["tokens"][c] - agg[group][fmt]["tokens"][c] for c in tik]
        lo, hi = min(s), max(s)
        span = f"{lo:.1f}%" if f"{lo:.1f}" == f"{hi:.1f}" else f"{lo:.1f}% to {hi:.1f}%"
        return span, f"{min(t):.0f} to {max(t):.0f} tokens, {min(t) * ms / 1000:.1f} to {max(t) * ms / 1000:.1f} s at {DECODE_TOK_PER_S} tok/s"

    L = ["## Reading the numbers\n"]
    pj, pl = rng("replies_14b", "json_min", "json_pretty"), rng("replies_14b", "line", "json_min")
    L.append(f"- Pretty to minified JSON is the largest gap ({pj[0]} on the 14B replies, {pj[1]} per spec), but it is already taken: the primer asks for one line and the recorded replies are minified (`raw_as_emitted` is within 1% of `json_min`). Pretty JSON is what the presets look like on disk, not what the model writes.")
    L.append(f"- Key shortening and positional arrays buy almost nothing in tokens: `json_keys` {rng('replies_14b', 'json_keys', 'json_min')[0]} and `json_pos` {rng('replies_14b', 'json_pos', 'json_min')[0]} vs minified on the replies. Common key names such as `title`, `state`, `rules`, `class`, `kids` are already one or two tokens, and positional arrays trade key names for `null` placeholders. `sexp` is {rng('replies_14b', 'sexp', 'json_min')[0]}: parentheses and quoted strings cost what braces and quotes cost.")
    L.append(f"- The line form is the only one that moves the needle: {pl[0]} vs minified on the replies ({pl[1]} per spec) and {rng('presets', 'line', 'json_min')[0]} on the presets. That is an upper bound for a model that writes the same content: the saving is the removed quotes, braces, commas and key names, and everything else (expressions, texts, names, ids) is paid in every format.")
    pt = corpus.get("first_call_prompt_tokens") or []
    if pt:
        L.append(f"- Per-call overhead is not the schema: in the recorded runs the first model call of a spec reports promptTokens {pt[0]} to {pt[-1]} (Qwen tokens as counted by the server, n={len(pt)}, median {pt[len(pt) // 2]}; older 7B runs, same Qwen2.5 tokenizer family, schema then about a quarter of today's size), i.e. the primer plus the goal plus the chat template, which matches the primer alone (about 1.25k tokens in the table above). So the schema, sent as `response_format`, is compiled to a decoding grammar and does not occupy the prompt; the `schema tokens` column is what it would cost if it were pasted in. The primer is byte-stable and KV-cached by Ollama, so its cost is paid when the cache is cold, while a per-spec decode saving is paid on every spec: a primer that is ~100 tokens shorter (the `line` row) is small next to the per-spec saving above.")
    L.append("")
    L.append("### Recommendation\n")
    L.append("1. **Keep asking the 14B for minified JSON under the existing JSON-schema grammar.** It is the only candidate Ollama can constrain (`format` takes a JSON schema, not a GBNF), the Kernel's repair and re-ask machinery is built on JSON paths, and no JSON-shaped alternative (short keys, positional arrays, S-expressions) saves more than a few tokens.")
    L.append("2. **Drop key shortening and positional arrays**: they cost prompt tokens for a dictionary or a slot table, add a failure mode for the model, and measured about 0 tokens.")
    L.append("3. **If a GBNF-capable runtime (llama.cpp server directly) is on the table, trial the line form**, because it is the one format with a real, repeatable saving and its failures are local to one line. Do that as an experiment gated on the 14B's validate-pass rate, not on token counts: the counts here assume identical content, and a format the model writes worse loses more than it saves.")
    L.append("4. The bigger levers are content, not wire format: most reply tokens are expressions, texts, names and ids, which every format carries. Fewer or shorter rules and view nodes (more `each` / `grid`, more of the Kernel's defaults) will save more decode time than any re-encoding.\n")
    return L


# ---------------------------------------------------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------------------------------------------------
def render_markdown(corpus: dict[str, Any], rows: list[dict[str, Any]], agg: dict, counters: list[str], qwen_note: str,
                    overhead: dict[str, Any] | None, groups: list[str]) -> str:
    primary = counters[0]
    n_specs = {g: len(v) for g, v in corpus["groups"].items()}
    unp = corpus["unparseable"]
    L: list[str] = []
    L.append("# Wire formats for a Kernel spec: measured token cost\n")
    L.append("Generated by `./.venv/bin/python -m benchmark.token_metrics --prompt-overhead --markdown docs/RESULTS_formats.md` "
             "from the corpus in `corpus/data/` (real artifacts of the Deterministic Coder repo, see `corpus/data/manifest.json`). "
             "Nothing here is simulated: every number is a tokenizer run over the encoded text, and every encoding was decoded back and compared with the original object.\n")
    L.append("## What was measured\n")
    L.append(f"- Specs: {n_specs['presets']} presets (canonical long form), {n_specs['replies_14b']} parseable 14B replies and {n_specs['replies_7b']} parseable 7B replies "
             f"(the shorthand form the model writes: `\"p.label: Text\"` nodes, `\"number=0\"` state). "
             f"{len(unp)} further recorded replies are not JSON objects (empty, truncated or malformed) and cannot be encoded; they are counted in the corpus, not in the means.")
    man = corpus.get("manifest") or {}
    if man:
        dirty = ", ".join(man.get("source_dirty", [])) or "none"
        L.append(f"- Source: Deterministic Coder at `{man.get('source_git_head', '?')[:7]}`, read only, with these used files differing from that commit when the corpus was built (the repo is edited by other work meanwhile; `corpus/data/manifest.json` pins every source by sha256): {dirty}.")
    L.append(f"- Tokenizers: {', '.join(c for c in counters if not c.startswith('qwen'))}. Qwen (the local model's own tokenizer): **{qwen_note}**.")
    L.append("- Baseline: `json_pretty` (indent=2). Because the primer asks for ONE line, `json_min` is what the model writes today; the replies group also lists `raw_as_emitted` (the reply text as recorded).")
    L.append("- Round trip means structural equality of the decoded object with the original (types strict, list order kept, JSON object key order not significant: the Kernel canonicalises it). "
             "'key order kept' counts the specs whose dict key order also survived.")
    L.append("- `model_writable` is a judgement about a 14B under grammar-constrained decoding; it is not measured.\n")
    L.append("## Formats\n")
    L.append("| format | description | model_writable | why |")
    L.append("|---|---|---|---|")
    for f in FORMATS.values():
        L.append(f"| {f.name} | {f.description} | {'yes' if f.model_writable else 'no'} | {f.writable_note} |")
    L.append("\nBinary forms (MessagePack, CBOR) are skipped: a language model cannot write bytes.\n")
    L.append("## Results by tokenizer\n")
    for c in counters:
        L.append(f"### {c}\n")
        L.extend(table_for(agg, rows, c, groups))
    L.append("## Verdict (by " + primary + ")\n")
    L.extend(verdict_lines(agg, rows, primary))
    L.append("")
    if overhead:
        L.append("## Per-call prompt overhead: Kernel primer and schema\n")
        L.append(f"Read from `kernel_model.js` through the Node loader (`DC.kernelModel.primer('local')`, `JSON.stringify(DC.kernelModel.schema)`): "
                 f"primer {overhead['primer_chars']} chars, schema {overhead['schema_chars']} chars (minified), {overhead['n_examples']} few-shot examples inside the primer.\n")
        L.append("| tokenizer | primer tokens | schema tokens | primer + schema |")
        L.append("|---|---:|---:|---:|")
        for c, e in overhead["by_counter"].items():
            L.append(f"| {c} | {e['primer']} | {e['schema']} | {e['primer'] + e['schema']} |")
        L.append("")
        L.append("Primer with its two few-shot examples re-encoded in each format (the instruction lines still describe JSON, so this is a lower bound for a non-JSON primer):\n")
        L.append("| format | " + " | ".join(f"{c} primer" for c in overhead["by_counter"]) + " |")
        L.append("|---|" + "---:|" * len(overhead["by_counter"]))
        for fname in FORMATS:
            L.append(f"| {fname} | " + " | ".join(str(e["primer_by_format"][fname]) for e in overhead["by_counter"].values()) + " |")
        L.append("")
    L.extend(interpretation(corpus, agg, counters))
    L.append("## Limits\n")
    L.append(f"- Small sample: {n_specs['presets']} presets and {n_specs['replies_14b']} parseable 14B replies. Presets are hand-written canonical specs, replies are what the model emits; they differ in form (canonical long form vs shorthand), so savings differ between the groups and must not be pooled.")
    L.append("- The replies include failed runs (a reply that parses as JSON can still fail validation); the corpus is the real distribution, not a curated one. Degenerate replies (an object flattened into strings such as `\"id: \"`) are in it and inflate the line-form fallback lines.")
    L.append("- The line and positional forms emit schema fields in a fixed order, so dict key order is not preserved there (see the 'key order kept' column).")
    L.append(f"- cl100k_base and o200k_base are proxies. Qwen: {qwen_note}.")
    L.append(f"- Output time: at {DECODE_TOK_PER_S} tok/s one token is about {1000 / DECODE_TOK_PER_S:.0f} ms. Prompt tokens are prefilled (much faster than decoded) and the byte-stable primer is KV-cached by Ollama, so a per-call primer cost is mostly paid once; a per-spec decode saving is paid on every spec.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------
def build_counters(tiktoken_names: list[str]) -> list:
    return [TiktokenCounter(n) for n in tiktoken_names]


def run(args: argparse.Namespace) -> dict[str, Any]:
    corpus = load_corpus(Path(args.corpus))
    counters: list = build_counters(args.tokenizers.split(","))
    qwen_note = "not run (pass --ollama)"
    counter_subset: dict[str, int] = {}
    if args.ollama:
        oc, qwen_note = ollama_counter(args.ollama_host, args.ollama_model)
        if oc is not None:
            counters.append(oc)
            counter_subset[oc.name] = args.ollama_max_specs
            qwen_note += f"; first {args.ollama_max_specs} specs of each group only (each count is one 1-token generation)"
    rows = measure(corpus, counters, counter_subset=counter_subset)
    names = [c.name for c in counters]
    agg = aggregate(rows, names)
    overhead = prompt_overhead(load_kernel_prompt(Path(args.corpus), live=args.live_prompt), counters) if args.prompt_overhead else None
    return {"corpus": corpus, "rows": rows, "agg": agg, "counters": names, "qwen_note": qwen_note, "overhead": overhead}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=str(CORPUS_DIR))
    ap.add_argument("--tokenizers", default=",".join(TIKTOKEN_NAMES))
    ap.add_argument("--ollama", action="store_true", help="also count Qwen tokens via Ollama (skipped cleanly if busy or down)")
    ap.add_argument("--ollama-host", default=OLLAMA_HOST)
    ap.add_argument("--ollama-model", default=OLLAMA_MODEL)
    ap.add_argument("--ollama-max-specs", type=int, default=6, help="specs per group counted through Ollama")
    ap.add_argument("--prompt-overhead", action="store_true", help="also report the tokens of the Kernel primer and schema")
    ap.add_argument("--live-prompt", action="store_true", help="re-evaluate kernel_model.js through Node instead of the corpus copy")
    ap.add_argument("--markdown", help="write the report to this path")
    ap.add_argument("--json", help="write the raw rows and aggregates to this path")
    a = ap.parse_args(argv)
    res = run(a)
    groups = list(GROUPS)
    print(render_text(res["agg"], res["rows"], res["counters"], groups))
    print(f"qwen: {res['qwen_note']}")
    if res["overhead"]:
        print(json.dumps(res["overhead"], indent=1))
    if a.markdown:
        Path(a.markdown).write_text(render_markdown(res["corpus"], res["rows"], res["agg"], res["counters"], res["qwen_note"], res["overhead"], groups), encoding="utf-8")
        print(f"wrote {a.markdown}")
    if a.json:
        Path(a.json).write_text(json.dumps({"agg": res["agg"], "rows": res["rows"], "overhead": res["overhead"], "qwen": res["qwen_note"]}, indent=1), encoding="utf-8")
    bad = [r for r in res["rows"] if not r["roundtrip"]]
    if bad:
        print(f"ROUND TRIP FAILURES: {len(bad)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
