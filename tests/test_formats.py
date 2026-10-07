"""Wire formats: exact round trips on the real corpus and on hostile random input, honest declarations, the token
counters, and a clean skip of the Ollama path."""
import os as _os
import pytest as _pytest
if not _os.path.exists(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "corpus/data/presets.json")):
    _pytest.skip("private corpus data is not published: regenerate it from a local deterministic-coder checkout (see README, Results so far)", allow_module_level=True)

import json
import random
import urllib.error
from pathlib import Path

import pytest

from benchmark import formats as F
from benchmark import token_metrics as tm

CORPUS = tm.load_corpus()
SPECS = [(g, it["id"], it["spec"]) for g, items in CORPUS["groups"].items() for it in items]
ORDER_KEPT = [n for n, f in F.FORMATS.items() if f.preserves_key_order]


def test_corpus_has_the_expected_specs():
    sizes = {g: len(v) for g, v in CORPUS["groups"].items()}
    assert sizes["presets"] == 6
    assert sizes["replies_14b"] == 15
    assert sizes["replies_7b"] == 16
    assert len(CORPUS["unparseable"]) == 17


@pytest.mark.parametrize("name", list(F.FORMATS))
def test_every_format_round_trips_every_corpus_spec(name):
    fmt = F.FORMATS[name]
    for g, sid, spec in SPECS:
        ok, ordered, text = fmt.roundtrip(spec)
        assert ok, f"{name} lost data on {g}/{sid}"
        if fmt.preserves_key_order:
            assert ordered, f"{name} reordered keys on {g}/{sid}"
        assert isinstance(text, str) and text


@pytest.mark.parametrize("name", list(F.FORMATS))
def test_encoding_is_deterministic(name):
    fmt = F.FORMATS[name]
    spec = SPECS[0][2]
    assert fmt.encode(spec) == fmt.encode(json.loads(json.dumps(spec)))


def test_declarations_are_complete():
    assert list(F.FORMATS) == ["json_pretty", "json_min", "json_keys", "json_pos", "line", "sexp"]
    for f in F.FORMATS.values():
        assert isinstance(f.model_writable, bool) and len(f.writable_note) > 20 and f.description


def test_equality_is_strict_about_types_and_lists():
    assert F.same({"a": 1, "b": 2}, {"b": 2, "a": 1})
    assert not F.same({"a": 1}, {"a": 1.0})
    assert not F.same({"a": 1}, {"a": True})
    assert not F.same([1, 2], [2, 1])
    assert not F.same_ordered({"a": 1, "b": 2}, {"b": 2, "a": 1})


# ---- hostile input: the formats must be exact for ANY JSON, not just the corpus -----------------------------------
WORDS = ["", " ", "a", "t", "s", "~t", "~", "tag", "kids", "class", "on", "do", "when", "then", "else", "if", "each", "set", "to",
         "push", "value", "title", "toggle", "view", "rules", "state", "?", "*", "#", "!", ">", "@", "|", "(%)", "(", ")", "%",
         "p.label: Text", "div#x.a.b", "count = count + 1", "x -> y", "a; b", "a & b", " lead", "trail ", "multi\nline", "tab\t",
         "quote\"s", "back\\slash", "{count}", "[a=b]", "null", "true", "12", "-1.5e3", "NaN", "é", "😊", "on: ", "id: ", "set x = 1",
         "title = 'x'", "x += 1", "click:#a, key:b", "number=0 persist", "@when", " @when c", "a, b", "~~a"]
KEYS = ["", "a", "t", "s", "~t", "tag", "id", "class", "text", "attrs", "style", "bind", "on", "kids", "size", "draw", "if", "then",
        "else", "each", "as", "index", "grid", "cell", "set", "to", "push", "value", "when", "do", "where", "with", "title", "state",
        "view", "rules", "theme", "~", "x y", "k:v", "é", "%", "@", "!"]


def rnd(rng, depth=0):
    r = rng.random()
    if depth > 3 or r < 0.35:
        return rng.choice([None, True, False, 0, 1, -7, 3.5, 1.0, 10**15, rng.choice(WORDS)])
    if r < 0.6:
        return [rnd(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rng.choice(KEYS): rnd(rng, depth + 1) for _ in range(rng.randint(0, 5))}


def rnd_spec(rng):
    """Spec-shaped but hostile: schema positions get random fields and random value types."""
    def node(d=0):
        r = rng.random()
        if d > 3 or r < 0.2:
            return rng.choice(WORDS + ["p.label: Hi", "button#a: Go"])
        n = {}
        pool = ["tag", "id", "class", "text", "attrs", "style", "bind", "on", "kids", "if", "then", "else", "each", "as", "index", "grid", "cell", "size", "draw", "t", "~x"]
        for k in rng.sample(pool, rng.randint(0, 5)):
            if k in ("kids", "then", "else"):
                n[k] = [node(d + 1) for _ in range(rng.randint(0, 3))]
            elif k == "cell":
                n[k] = node(d + 1)
            elif k == "on":
                n[k] = {rng.choice(["click", "key:Enter", "x y"]): rng.choice([[effect() for _ in range(rng.randint(0, 2))], {"when": "a", "do": [effect()]}, "oops"]) for _ in range(rng.randint(0, 2))}
            elif k in ("attrs", "style"):
                n[k] = {rng.choice(["type", "aria-label", "x=y", "a b"]): rng.choice(WORDS) for _ in range(rng.randint(0, 3))}
            elif k == "grid":
                n[k] = rng.choice([[3, 3], [1], "x", [True, 2]])
            else:
                n[k] = rng.choice(WORDS + [1, None, [], {}])
        return n

    def effect():
        e = {}
        for k in rng.sample(["set", "to", "push", "value", "toggle", "clear", "removeWhere", "where", "updateWhere", "with", "start", "stop",
                             "reset", "announce", "title", "when", "zzz"], rng.randint(0, 4)):
            e[k] = rng.choice(WORDS + [{"a": "1", "b b": "2"}, {}, 4, None])
        return e

    s = {}
    for k in rng.sample(["v", "title", "layout", "state", "derived", "fns", "view", "time", "rules", "expose", "theme", "weird key", "t", "~t"], rng.randint(0, 9)):
        if k == "view":
            s[k] = [node() for _ in range(rng.randint(0, 4))]
        elif k == "rules":
            s[k] = [rng.choice([{"on": rng.choice(["init", "click:#a", ["click:#a", "key:b"], ["x"], 3]), "do": [effect() for _ in range(rng.randint(0, 3))]},
                                {"on": "init", "when": rng.choice(WORDS), "do": [effect()]}, "x", {}]) for _ in range(rng.randint(0, 4))]
        elif k in ("state", "derived", "fns", "expose", "time"):
            s[k] = {rng.choice(KEYS + ["count", "items"]): rng.choice(WORDS + [{"type": "number", "init": 0, "persist": True}, 5, None]) for _ in range(rng.randint(0, 3))}
        else:
            s[k] = rng.choice(WORDS + [{"mode": "dark", "accent": "#fff"}, 1, None])
    return s


@pytest.mark.parametrize("name", list(F.FORMATS))
def test_exact_on_random_json(name):
    rng = random.Random(1234)
    fmt = F.FORMATS[name]
    for i in range(400):
        v = rnd(rng) if i % 2 else rnd_spec(rng)
        ok, _, text = fmt.roundtrip(v)
        assert ok, f"{name} not exact on {json.dumps(v)[:300]!r} -> {text[:300]!r}"


@pytest.mark.parametrize("name", list(F.FORMATS))
def test_exact_on_random_specs(name):
    rng = random.Random(99)
    fmt = F.FORMATS[name]
    for _ in range(600):
        v = rnd_spec(rng)
        ok, _, text = fmt.roundtrip(v)
        assert ok, f"{name} not exact on {json.dumps(v)[:300]!r} -> {text[:300]!r}"


def test_line_form_is_one_rule_per_line_and_reads_like_the_example():
    spec = {"title": "T", "state": {"count": "number=0"}, "view": ["p.label: Count"],
            "rules": [{"on": "click:#inc", "do": [{"set": "count", "to": "count + 1"}]}]}
    text = F.FORMATS["line"].encode(spec)
    assert "  on click:#inc -> count = count + 1" in text.split("\n")
    assert "  p.label: Count" in text.split("\n")


def test_decoders_reject_garbage_with_formaterror():
    for name in ("json_keys", "json_pos", "sexp", "line"):
        with pytest.raises((F.FormatError, ValueError)):
            F.FORMATS[name].decode("(% a")  if name == "sexp" else F.FORMATS[name].decode("{not json" if name != "line" else "rules:\n  nonsense")


# ---- token counters ----------------------------------------------------------------------------------------------
def test_tiktoken_counter_is_deterministic_and_correct():
    c = tm.TiktokenCounter("cl100k_base")
    assert c.count("hello world") == 2
    text = F.FORMATS["json_min"].encode(SPECS[0][2])
    assert c.count(text) == c.count(text) > 100
    assert tm.TiktokenCounter("o200k_base").count("") == 0


def test_measure_and_aggregate_are_deterministic_and_json_min_saves():
    counters = [tm.TiktokenCounter("cl100k_base")]
    sub = {"groups": {"presets": CORPUS["groups"]["presets"][:2], "replies_14b": CORPUS["groups"]["replies_14b"][:2]},
           "unparseable": [], "prompts": []}
    r1, r2 = tm.measure(sub, counters), tm.measure(sub, counters)
    assert r1 == r2
    assert len(r1) == 2 * len(F.FORMATS) + 2 * (len(F.FORMATS) + 1)
    agg = tm.aggregate(r1, ["cl100k_base"])
    p = agg["presets"]
    assert p["json_min"]["tokens"]["cl100k_base"] < p["json_pretty"]["tokens"]["cl100k_base"]
    assert all(a["ok"] == a["n"] for g in agg.values() for a in g.values())
    assert tm.saving(60, 100) == pytest.approx(40.0)


def test_prompt_overhead_counts_primer_and_schema():
    kp = tm.load_kernel_prompt()
    ov = tm.prompt_overhead(kp, [tm.TiktokenCounter("cl100k_base")])
    e = ov["by_counter"]["cl100k_base"]
    assert ov["n_examples"] == 2
    assert e["primer"] > 500 and e["schema"] > 500
    assert e["primer_by_format"]["json_min"] == e["primer"]  # the examples are already minified JSON
    assert e["primer_by_format"]["json_pretty"] > e["primer"]


# ---- the Ollama path ---------------------------------------------------------------------------------------------
def args(**kw):
    ns = dict(corpus=str(tm.CORPUS_DIR), tokenizers="cl100k_base", ollama=False, ollama_host="http://127.0.0.1:1", ollama_model="m",
              ollama_max_specs=2, prompt_overhead=False, live_prompt=False, markdown=None, json=None)
    ns.update(kw)
    import argparse
    return argparse.Namespace(**ns)


def test_no_http_without_the_flag(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("HTTP call without --ollama")
    monkeypatch.setattr(tm, "_http_json", boom)
    res = tm.run(args())
    assert res["qwen_note"].startswith("not run")
    assert res["counters"] == ["cl100k_base"]


def test_ollama_unreachable_is_skipped_cleanly(monkeypatch):
    def down(*a, **k):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(tm, "_http_json", down)
    counter, note = tm.ollama_counter("http://127.0.0.1:1")
    assert counter is None and note.startswith("skipped") and "not reachable" in note
    res = tm.run(args(ollama=True))
    assert res["qwen_note"].startswith("skipped")
    assert res["counters"] == ["cl100k_base"]
    assert all(set(r["tokens"]) == {"cl100k_base"} for r in res["rows"])


def test_ollama_probe_failure_or_slowness_is_skipped(monkeypatch):
    def ps_ok_probe_times_out(url, payload=None, timeout=10.0):
        if payload is None:
            return {"models": [{"name": "m"}]}
        raise TimeoutError("busy")
    monkeypatch.setattr(tm, "_http_json", ps_ok_probe_times_out)
    counter, note = tm.ollama_counter("http://x")
    assert counter is None and "probe failed or was slow" in note
    monkeypatch.setattr(tm, "_http_json", lambda url, payload=None, timeout=10.0: {"models": []} if payload is None else {"prompt_eval_count": 1})
    counter, note = tm.ollama_counter("http://x", probe_s=-1.0)
    assert counter is None and "probe took" in note


def test_ollama_counts_subtract_the_nonce_and_use_distinct_prefixes(monkeypatch):
    seen = []

    def fake(url, payload=None, timeout=10.0):
        if payload is None:
            return {"models": []}
        seen.append(payload["prompt"])
        assert payload["raw"] is True and payload["options"]["num_predict"] == 1 and payload["stream"] is False
        nonce, _, body = payload["prompt"].partition("\n")
        return {"prompt_eval_count": len(nonce) + 1 + len(body)}  # one 'token' per character, nonce included

    monkeypatch.setattr(tm, "_http_json", fake)
    counter, note = tm.ollama_counter("http://x")
    assert counter is not None and note.startswith("measured")
    assert counter.count("abc") == 3 and counter.count("") == 0
    assert len({p[0] for p in seen[-3:]}) == 3  # consecutive prompts never share their first character

    res = tm.run(args(ollama=True))
    q = counter.name
    assert q in res["counters"] and "first 2 specs" in res["qwen_note"]
    capped = [r for r in res["rows"] if q in r["tokens"]]
    assert {r["group"] for r in capped} == {"presets", "replies_14b", "replies_7b"}
    assert len({r["id"] for r in capped if r["group"] == "presets"}) == 2
    # the fake tokenizer counts characters, so every Qwen count equals the format's character count
    assert all(r["tokens"][q] == r["chars"] for r in capped)


def test_cli_writes_the_markdown_report(tmp_path):
    out = tmp_path / "r.md"
    assert tm.main(["--tokenizers", "cl100k_base", "--prompt-overhead", "--markdown", str(out)]) == 0
    md = out.read_text(encoding="utf-8")
    for heading in ("## Formats", "### cl100k_base", "## Verdict", "## Per-call prompt overhead", "## Limits"):
        assert heading in md
    assert "not run (pass --ollama)" in md
