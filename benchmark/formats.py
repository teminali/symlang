"""Alternative wire formats for a Kernel app spec (a JSON object), all lossless.

The Kernel spec is what a small local LLM WRITES (output tokens are the bottleneck), so the question is which
text form of the same object costs the fewest tokens and still parses back to exactly the same object.

Every format here is `encode(obj) -> str` / `decode(str) -> obj` with a real parser, and is exact for any JSON
value, not just the corpus: where a schema-aware shortcut does not apply (an irregular node, an unknown key, an
ambiguous string) the encoder falls back, locally and verified, to inline JSON for just that piece.

"Same object" means structural equality: JSON objects are unordered (the Kernel itself canonicalises key order,
see kernel_core.js), lists keep their order, and bool / int / float / str / null are never confused. Formats that
also keep the textual key order declare `preserves_key_order`; the positional and line forms emit schema fields in
a fixed order instead.

  json_pretty  indent=2 JSON (what a canonical spec looks like on disk)
  json_min     minified JSON, what the primer asks the model to write today ("on a single line")
  json_keys    minified JSON with a fixed key dictionary (state->s, rules->r, ...), applied by schema position
  json_pos     minified JSON where rules, effects and view nodes are positional arrays (no key names)
  line         one rule / view node per line, `on click:#inc -> count = clamp(count + 1, 0, 9)`
  sexp         S-expression: `(% title "Counter" rules ((% on "click:#inc" ...)))`

MessagePack / CBOR-like binary forms are skipped on purpose: a language model cannot write bytes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["Format", "FormatError", "FORMATS", "get_format", "same", "same_ordered", "dumps_min"]


class FormatError(ValueError):
    """Raised by a decoder on text it cannot read."""


# ---------------------------------------------------------------------------------------------------------------------
# equality: strict about JSON types, not about key order
# ---------------------------------------------------------------------------------------------------------------------
def _kind(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def same(a: Any, b: Any) -> bool:
    """Structural JSON equality: types strict (1 != 1.0 != True), lists ordered, dict key order ignored."""
    ka = _kind(a)
    if ka != _kind(b):
        return False
    if ka == "list":
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if ka == "dict":
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def same_ordered(a: Any, b: Any) -> bool:
    """`same`, and every dict also has the same key order."""
    ka = _kind(a)
    if ka != _kind(b):
        return False
    if ka == "list":
        return len(a) == len(b) and all(same_ordered(x, y) for x, y in zip(a, b))
    if ka == "dict":
        return list(a) == list(b) and all(same_ordered(a[k], b[k]) for k in a)
    return a == b


def dumps_min(v: Any) -> str:
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


_JSON = json.JSONDecoder()


def _raw_json(text: str, pos: int = 0):
    """Parse one JSON value starting exactly at `pos`; returns (value, end)."""
    try:
        return _JSON.raw_decode(text, pos)
    except ValueError as e:  # includes JSONDecodeError
        raise FormatError(f"bad JSON at offset {pos}: {e}") from None


# ---------------------------------------------------------------------------------------------------------------------
# the spec shapes (positions in a spec where an object has known fields)
# ---------------------------------------------------------------------------------------------------------------------
# A child spec is: None (leaf, left alone) | "shape" | ("L", child) list | ("M", child) user-keyed map | ("U", a, b) either.
# Anything that does not match the declared structure is left verbatim, in both directions.
SHAPES: dict[str, dict[str, Any]] = {
    "spec": {
        "keys": {"v": "v", "title": "t", "lang": "l", "layout": "y", "storageKey": "k", "theme": "h", "state": "s",
                 "derived": "d", "fns": "f", "view": "w", "time": "m", "rules": "r", "expose": "x"},
        "kids": {"theme": "theme", "state": ("M", "sentry"), "time": ("M", "timer"), "view": ("L", "node"),
                 "rules": ("L", "rule")},
    },
    "theme": {"keys": {"mode": "m", "accent": "a"}, "kids": {}},
    "sentry": {"keys": {"type": "t", "init": "i", "persist": "p"}, "kids": {}},
    "timer": {"keys": {"every": "e", "after": "a", "while": "w"}, "kids": {}},
    "rule": {"keys": {"on": "o", "when": "w", "do": "d"}, "kids": {"do": ("L", "effect")}},
    "handler": {"keys": {"when": "w", "do": "d"}, "kids": {"do": ("L", "effect")}},
    "effect": {
        "keys": {"set": "s", "to": "t", "push": "p", "value": "v", "toggle": "g", "clear": "c", "removeWhere": "x",
                 "where": "w", "updateWhere": "u", "with": "h", "start": "a", "stop": "o", "reset": "z",
                 "announce": "n", "title": "i", "when": "e"},
        "kids": {"set": ("M", None), "push": ("M", None), "with": ("M", None)},
    },
    "node": {
        "keys": {"tag": "g", "id": "i", "class": "c", "text": "t", "attrs": "a", "style": "y", "bind": "b", "on": "o",
                 "kids": "k", "size": "z", "draw": "d", "autofocus": "f", "if": "q", "then": "h", "else": "e",
                 "each": "x", "as": "s", "index": "n", "grid": "r", "cell": "l"},
        "kids": {"attrs": ("M", None), "style": ("M", None), "on": ("M", ("U", ("L", "effect"), "handler")),
                 "kids": ("L", "node"), "then": ("L", "node"), "else": ("L", "node"), "cell": "node"},
    },
}
for _name, _s in SHAPES.items():  # the short codes must be a bijection per shape
    assert len(set(_s["keys"].values())) == len(_s["keys"]), _name
    _s["rev"] = {v: k for k, v in _s["keys"].items()}


# ---------------------------------------------------------------------------------------------------------------------
# json_keys: fixed key dictionary
# ---------------------------------------------------------------------------------------------------------------------
def _walk_keys(v: Any, spec: Any, dec: bool) -> Any:
    if spec is None:
        return v
    if isinstance(spec, tuple):
        tag = spec[0]
        if tag == "L":
            return [_walk_keys(x, spec[1], dec) for x in v] if isinstance(v, list) else v
        if tag == "M":
            return {k: _walk_keys(x, spec[1], dec) for k, x in v.items()} if isinstance(v, dict) else v
        if tag == "U":
            for alt in spec[1:]:
                if isinstance(alt, tuple):  # an ("L", ..) alternative matches lists
                    if isinstance(v, list):
                        return _walk_keys(v, alt, dec)
                elif isinstance(v, dict):
                    return _walk_keys(v, alt, dec)
            return v
        raise AssertionError(spec)
    if not isinstance(v, dict):
        return v
    shape = SHAPES[spec]
    fwd, rev, kids = shape["keys"], shape["rev"], shape["kids"]
    out = {}
    for k, x in v.items():
        if dec:
            if k in rev:
                long = rev[k]
                out[long] = _walk_keys(x, kids.get(long), True)
            elif k.startswith("~"):
                out[k[1:]] = x
            else:
                out[k] = x
        elif k in fwd:
            out[fwd[k]] = _walk_keys(x, kids.get(k), False)
        else:  # an unmapped key; escape it when it would be read back as a code
            out["~" + k if (k in rev or k.startswith("~")) else k] = x
    return out


def _keys_encode(obj: Any) -> str:
    return dumps_min(_walk_keys(obj, "spec", False))


def _keys_decode(text: str) -> Any:
    try:
        return _walk_keys(json.loads(text), "spec", True)
    except ValueError as e:
        raise FormatError(str(e)) from None


# ---------------------------------------------------------------------------------------------------------------------
# json_pos: positional arrays for rules, effects and view nodes
# ---------------------------------------------------------------------------------------------------------------------
_RULE_FIELDS = ["on", "when", "do"]
_VERBS = ["set", "push", "toggle", "clear", "removeWhere", "updateWhere", "start", "stop", "reset", "announce", "title"]
_EFFECT_LAYOUT = {
    "set": ["to", "when"], "push": ["value", "when"], "toggle": ["when"], "clear": ["when"],
    "removeWhere": ["where", "when"], "updateWhere": ["where", "with", "when"], "start": ["when"], "stop": ["when"],
    "reset": ["when"], "announce": ["when"], "title": ["when"],
}
_NODE_PLAIN = ["tag", "id", "class", "text", "attrs", "style", "bind", "on", "kids", "size", "draw", "autofocus"]
_NODE_VARIANTS = {"?": ["if", "then", "else"], "*": ["each", "as", "index", "kids"], "#": ["grid", "cell", "id", "class"]}
_VARIANT_OF = {"if": "?", "each": "*", "grid": "#"}

_POS_CHILD: dict[str, dict[str, Any]] = {
    "rule": {"do": ("L", "effect")},
    "effect": {},
    "node": {"on": ("M", ("U", ("L", "effect"), "handler")), "kids": ("L", "node"), "then": ("L", "node"),
             "else": ("L", "node"), "cell": "node"},
}


def _pos_layout(shape: str, d: dict) -> tuple[str | None, list[str]] | None:
    """(discriminator, ordered fields) when `d` is regular for the positional form of `shape`, else None."""
    if any(x is None for x in d.values()):
        return None
    keys = set(d)
    if shape == "rule":
        return (None, _RULE_FIELDS) if keys <= set(_RULE_FIELDS) else None
    if shape == "effect":
        verbs = [k for k in d if k in _EFFECT_LAYOUT]
        if len(verbs) != 1:
            return None
        verb = verbs[0]
        return (verb, _EFFECT_LAYOUT[verb]) if keys <= {verb, *_EFFECT_LAYOUT[verb]} else None
    if shape == "node":
        for key, sig in _VARIANT_OF.items():
            if key in d:
                return (sig, _NODE_VARIANTS[sig]) if keys <= set(_NODE_VARIANTS[sig]) else None
        if isinstance(d.get("tag"), str) and d["tag"] in _NODE_VARIANTS:
            return None
        return (None, _NODE_PLAIN) if keys <= set(_NODE_PLAIN) else None
    raise AssertionError(shape)


def _pos(v: Any, spec: Any, dec: bool) -> Any:
    if spec is None:
        return v
    if isinstance(spec, tuple):
        tag = spec[0]
        if tag == "L":
            return [_pos(x, spec[1], dec) for x in v] if isinstance(v, list) else v
        if tag == "M":
            return {k: _pos(x, spec[1], dec) for k, x in v.items()} if isinstance(v, dict) else v
        if tag == "U":
            for alt in spec[1:]:
                if isinstance(alt, tuple):
                    if isinstance(v, list):
                        return _pos(v, alt, dec)
                elif isinstance(v, dict):
                    return _pos(v, alt, dec)
            return v
        raise AssertionError(spec)
    if spec in ("rule", "effect", "node"):
        return _pos_dec(v, spec) if dec else _pos_enc(v, spec)
    # container shapes (spec, handler): keys stay, children are transformed
    if not isinstance(v, dict):
        return v
    kids = {"spec": {"state": None, "view": ("L", "node"), "rules": ("L", "rule")},
            "handler": {"do": ("L", "effect")}}[spec]
    return {k: _pos(x, kids.get(k), dec) for k, x in v.items()}


def _pos_enc(v: Any, shape: str) -> Any:
    if isinstance(v, str):
        return v
    if not isinstance(v, dict):
        return {"~": v}
    lay = _pos_layout(shape, v)
    if lay is None:
        return {"~": v} if "~" in v else v
    disc, fields = lay
    child = _POS_CHILD[shape]
    vals = [_pos(v[f], child.get(f), False) if f in v else None for f in fields]
    if shape == "effect":
        row, keep = [disc, v[disc]] + vals, 2
    elif shape == "node" and disc is not None:
        row, keep = [disc] + vals, 1
    else:
        row, keep = vals, 0
    while len(row) > keep and row[-1] is None:
        row.pop()
    return row


def _pos_dec(v: Any, shape: str) -> Any:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v["~"] if list(v) == ["~"] else v
    if not isinstance(v, list):
        return v
    child = _POS_CHILD[shape]
    out: dict[str, Any] = {}
    if shape == "rule":
        for f, x in zip(_RULE_FIELDS, v):
            if x is not None:
                out[f] = _pos(x, child.get(f), True)
        return out
    if shape == "effect":
        verb = v[0]
        out[verb] = v[1] if len(v) > 1 else None
        for f, x in zip(_EFFECT_LAYOUT[verb], v[2:]):
            if x is not None:
                out[f] = _pos(x, child.get(f), True)
        return out
    # node
    head = v[0] if v else None
    if isinstance(head, str) and head in _NODE_VARIANTS:
        fields, vals = _NODE_VARIANTS[head], v[1:]
    else:
        fields, vals = _NODE_PLAIN, v
    for f, x in zip(fields, vals):
        if x is not None:
            out[f] = _pos(x, child.get(f), True)
    return out


def _pos_encode(obj: Any) -> str:
    return dumps_min(_pos(obj, "spec", False))


def _pos_decode(text: str) -> Any:
    try:
        return _pos(json.loads(text), "spec", True)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise FormatError(f"bad positional JSON: {e}") from None


# ---------------------------------------------------------------------------------------------------------------------
# sexp: (% k v k v) is an object, (a b c) a list, "text" a string, bare 12 true false null
# ---------------------------------------------------------------------------------------------------------------------
_SYM = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*\Z")
_BARE_WORDS = {"true", "false", "null"}


def _sx(v: Any, out: list[str]) -> None:
    k = _kind(v)
    if k == "null":
        out.append("null")
    elif k == "bool":
        out.append("true" if v else "false")
    elif k in ("int", "float"):
        out.append(json.dumps(v))
    elif k == "str":
        out.append(json.dumps(v, ensure_ascii=False))
    elif k == "list":
        out.append("(")
        for i, x in enumerate(v):
            if i:
                out.append(" ")
            _sx(x, out)
        out.append(")")
    elif k == "dict":
        out.append("(%")
        for key, x in v.items():
            out.append(" ")
            out.append(key if _SYM.match(key) and key not in _BARE_WORDS else json.dumps(key, ensure_ascii=False))
            out.append(" ")
            _sx(x, out)
        out.append(")")
    else:
        raise FormatError(f"not JSON: {type(v).__name__}")


def _sexp_encode(obj: Any) -> str:
    out: list[str] = []
    _sx(obj, out)
    return "".join(out)


_SX_TOKEN = re.compile(r'\s*(?:(\()|(\))|("(?:[^"\\]|\\.)*")|([^\s()"]+))', re.S)


def _sexp_decode(text: str) -> Any:
    pos = 0
    n = len(text)

    def tok():
        nonlocal pos
        m = _SX_TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == "":
                return None
            raise FormatError(f"unreadable S-expression at offset {pos}")
        pos = m.end()
        if m.group(1):
            return ("(", None)
        if m.group(2):
            return (")", None)
        if m.group(3):
            return ("s", json.loads(m.group(3)))
        word = m.group(4)
        return ("a", word)

    def atom(word):
        if word in _BARE_WORDS:
            return {"true": True, "false": False, "null": None}[word]
        try:
            return json.loads(word)
        except ValueError:
            raise FormatError(f"unknown atom {word!r}") from None

    def value(t):
        if t is None:
            raise FormatError("unexpected end of input")
        kind, val = t
        if kind == "s":
            return val
        if kind == "a":
            return atom(val)
        if kind == ")":
            raise FormatError("unbalanced ')'")
        # "("
        first = tok()
        if first is not None and first == ("a", "%"):
            obj = {}
            while True:
                t = tok()
                if t is None:
                    raise FormatError("unterminated object")
                if t[0] == ")":
                    return obj
                if t[0] == "s":
                    key = t[1]
                elif t[0] == "a" and _SYM.match(t[1]):
                    key = t[1]
                else:
                    raise FormatError(f"bad object key {t!r}")
                obj[key] = value(tok())
        items = []
        t = first
        while True:
            if t is None:
                raise FormatError("unterminated list")
            if t[0] == ")":
                return items
            items.append(value(t))
            t = tok()

    v = value(tok())
    if pos < n and text[pos:].strip():
        raise FormatError("trailing text after the S-expression")
    return v


# ---------------------------------------------------------------------------------------------------------------------
# line: one rule / view node per line
# ---------------------------------------------------------------------------------------------------------------------
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_KEYNAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*\Z")
_HEAD = re.compile(r"(?P<tag>[a-z][a-z0-9]*)?(?:#(?P<id>[A-Za-z0-9_{}\-]+))?(?P<cls>(?:\.[A-Za-z0-9_{}\-]+)*)")
_EVENT = re.compile(r"[^\s]+\Z")
_INDENT = "  "


def _tok_enc(v: Any) -> str:
    """A value in a `name: value` slot: a raw string when that is unambiguous, otherwise inline JSON."""
    if isinstance(v, str) and v and v == v.strip() and "\n" not in v and "\r" not in v and v[0] not in '"{[':
        try:
            json.loads(v)
        except ValueError:
            return v
    return dumps_min(v)


def _tok_dec(s: str) -> Any:
    if s[:1] in ('"', "{", "["):
        v, end = _raw_json(s, 0)
        if s[end:].strip():
            raise FormatError(f"trailing text after JSON value: {s[end:end + 20]!r}")
        return v
    try:
        return json.loads(s)
    except ValueError:
        return s


def _text_enc(s: str) -> str:
    """The text slot at the end of a node line: it is known to be a string, so a raw remainder is unambiguous."""
    if s and s == s.strip() and "\n" not in s and "\r" not in s and s[0] != '"':
        return s
    return json.dumps(s, ensure_ascii=False)


def _text_dec(s: str) -> str:
    if s.startswith('"'):
        v, end = _raw_json(s, 0)
        if s[end:] or not isinstance(v, str):
            raise FormatError("bad quoted text")
        return v
    return s


def _is_str(x: Any) -> bool:
    return isinstance(x, str)


def _key_enc(k: str) -> str:
    return k if _KEYNAME.match(k) else json.dumps(k, ensure_ascii=False)


def _key_dec(line: str) -> tuple[str, str]:
    """Split `key: rest` where key is a bare name or a JSON string. Returns (key, rest)."""
    if line.startswith('"'):
        k, end = _raw_json(line, 0)
        if line[end:end + 2] != ": " and line[end:] != ":":
            raise FormatError(f"expected ': ' after key in {line[:40]!r}")
        return k, line[end + 2:]
    i = line.find(": ")
    if i < 0:
        if line.endswith(":"):
            return line[:-1], ""
        raise FormatError(f"expected 'name: value' in {line[:40]!r}")
    return line[:i], line[i + 2:]


# ---- effects ---------------------------------------------------------------------------------------------------------
_WORD_VERBS = ["toggle", "clear", "start", "stop", "reset", "announce", "title"]


def _pairs_enc(m: dict, op: str) -> str | None:
    if not m or not all(_IDENT.match(k) and _is_str(x) for k, x in m.items()):
        return None
    return " & ".join(f"{k} {op} {x}" for k, x in m.items())


def _pairs_dec(s: str, op: str) -> dict:
    out = {}
    for part in s.split(" & "):
        k, sep, x = part.partition(f" {op} ")
        if not sep:
            raise FormatError(f"expected '{op}' in {part[:40]!r}")
        out[k] = x
    return out


def _effect_term(e: Any) -> str | None:
    """The sugar for one effect dict, or None. (Verified by the caller.)"""
    if not isinstance(e, dict):
        return None
    keys = list(e)
    when = e.get("when")
    if "when" in e and not _is_str(when):
        return None
    body = None
    rest = {k for k in keys if k != "when"}
    if "set" in e:
        s = e["set"]
        if isinstance(s, str) and rest == {"set", "to"} and _is_str(e["to"]) and _IDENT.match(s):
            body = f"{s} = {e['to']}"
        elif isinstance(s, dict) and rest == {"set"}:
            p = _pairs_enc(s, "=")
            body = None if p is None else f"set {p}"
    elif "push" in e:
        s = e["push"]
        if isinstance(s, str) and rest == {"push", "value"} and _is_str(e["value"]) and _IDENT.match(s):
            body = f"{s} += {e['value']}"
        elif isinstance(s, dict) and rest == {"push"}:
            p = _pairs_enc(s, "=")
            body = None if p is None else f"push {p}"
    elif "removeWhere" in e:
        if rest == {"removeWhere", "where"} and _is_str(e["removeWhere"]) and _is_str(e["where"]):
            body = f"removeWhere {e['removeWhere']} where {e['where']}"
    elif "updateWhere" in e:
        w = e.get("with")
        if rest == {"updateWhere", "where", "with"} and _is_str(e["updateWhere"]) and _is_str(e["where"]) and isinstance(w, dict):
            p = _pairs_enc(w, "=")
            body = None if p is None else f"updateWhere {e['updateWhere']} where {e['where']} with {p}"
    else:
        for verb in _WORD_VERBS:
            if verb in e:
                if rest == {verb} and _is_str(e[verb]) and e[verb]:
                    body = f"{verb} {e[verb]}"
                break
    if body is None:
        return None
    return body + (f" @when {when}" if "when" in e else "")


def _effect_parse_term(t: str) -> dict:
    when = None
    has_when = False
    i = t.find(" @when ")
    if i >= 0:
        t, when, has_when = t[:i], t[i + 7:], True
    out: dict[str, Any]
    head, _, tail = t.partition(" ")
    if tail.startswith("= ") or tail.startswith("+= "):  # `title = ..` is an assignment, not the verb `title`
        head = ""
    if head == "set" and tail and "=" in tail and not tail.startswith("="):
        out = {"set": _pairs_dec(tail, "=")}
    elif head == "push" and tail and "=" in tail and not tail.startswith("="):
        out = {"push": _pairs_dec(tail, "=")}
    elif head == "removeWhere":
        lst, sep, where = tail.partition(" where ")
        if not sep:
            raise FormatError(f"bad removeWhere: {t[:40]!r}")
        out = {"removeWhere": lst, "where": where}
    elif head == "updateWhere":
        lst, sep, rest = tail.partition(" where ")
        where, sep2, w = rest.partition(" with ")
        if not sep or not sep2:
            raise FormatError(f"bad updateWhere: {t[:40]!r}")
        out = {"updateWhere": lst, "where": where, "with": _pairs_dec(w, "=")}
    elif head in _WORD_VERBS:
        out = {head: tail}
    else:
        name, sep, expr = t.partition(" += ")
        if sep:
            out = {"push": name, "value": expr}
        else:
            name, sep, expr = t.partition(" = ")
            if not sep:
                raise FormatError(f"unreadable effect: {t[:40]!r}")
            out = {"set": name, "to": expr}
    if has_when:
        out["when"] = when
    return out


def _effects_enc(effs: list) -> str:
    parts = []
    for e in effs:
        term = _effect_term(e)
        if term is not None:
            try:
                ok = same(_effect_parse_term(term), e) and "; " not in term
            except FormatError:
                ok = False
            if ok and "\n" not in term and "\r" not in term:
                parts.append(term)
                continue
        parts.append("!" + dumps_min(e))
    return "; ".join(parts)


def _effects_dec(s: str) -> list:
    out: list = []
    pos = 0
    n = len(s)
    if not s:
        return out
    while True:
        if s.startswith("!", pos):
            v, pos = _raw_json(s, pos + 1)
            out.append(v)
        else:
            j = s.find("; ", pos)
            if j < 0:
                j = n
            out.append(_effect_parse_term(s[pos:j]))
            pos = j
        if pos >= n:
            return out
        if s.startswith("; ", pos):
            pos += 2
        else:
            raise FormatError(f"expected '; ' between effects at {s[pos:pos + 20]!r}")


# ---- rules -----------------------------------------------------------------------------------------------------------
def _rule_line(r: Any) -> str:
    if isinstance(r, dict) and set(r) <= {"on", "when", "do"} and "on" in r and isinstance(r.get("do"), list) and r["do"]:
        on = r["on"]
        trig = None
        if _is_str(on) and on:
            trig = on
        elif isinstance(on, list) and len(on) >= 2 and all(_is_str(x) and x for x in on):
            trig = ", ".join(on)
        when = r.get("when")
        if trig is not None and ("when" not in r or (_is_str(when) and when)):
            line = f"on {trig}" + (f" when {when}" if "when" in r else "") + " -> " + _effects_enc(r["do"])
            try:
                if same(_rule_parse(line), r) and "\n" not in line and "\r" not in line:
                    return line
            except FormatError:
                pass
    return "!" + dumps_min(r)


def _rule_parse(line: str) -> Any:
    if line.startswith("!"):
        v, end = _raw_json(line, 1)
        if line[end:]:
            raise FormatError("trailing text after !json rule")
        return v
    if not line.startswith("on "):
        raise FormatError(f"a rule line starts with 'on ': {line[:40]!r}")
    head, sep, effs = line[3:].partition(" -> ")
    if not sep:
        raise FormatError(f"a rule line needs ' -> ': {line[:40]!r}")
    trig, wsep, when = head.partition(" when ")
    rule: dict[str, Any] = {"on": trig.split(", ") if ", " in trig else trig}
    if wsep:
        rule["when"] = when
    rule["do"] = _effects_dec(effs)
    return rule


# ---- view nodes ------------------------------------------------------------------------------------------------------
_SIMPLE = r"[A-Za-z0-9_{}\-]+"


def _head_enc(n: dict) -> tuple[str, set[str]]:
    """The `tag#id.class` head for the parts of `n` that fit it, and the keys that did not (they go to the extras)."""
    h, left = "", set()
    tag, id_, cls = n.get("tag"), n.get("id"), n.get("class")
    if "tag" in n:
        if _is_str(tag) and re.fullmatch(r"[a-z][a-z0-9]*", tag):
            h += tag
        else:
            left.add("tag")
    if "id" in n:
        if _is_str(id_) and re.fullmatch(_SIMPLE, id_):
            h += "#" + id_
        else:
            left.add("id")
    if "class" in n:
        if _is_str(cls) and re.fullmatch(_SIMPLE + "(?: " + _SIMPLE + ")*", cls):
            h += "".join("." + c for c in cls.split(" "))
        else:
            left.add("class")
    return h, left


def _head_dec(h: str) -> dict:
    m = _HEAD.fullmatch(h)
    if not m:
        raise FormatError(f"bad node head {h!r}")
    out = {}
    if m.group("tag"):
        out["tag"] = m.group("tag")
    if m.group("id"):
        out["id"] = m.group("id")
    if m.group("cls"):
        out["class"] = m.group("cls")[1:].replace(".", " ")
    return out


def _attrs_enc(a: dict) -> str | None:
    if not a or not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_:\-]*", k) and _is_str(x) for k, x in a.items()):
        return None
    return "[" + ", ".join(f"{k}={x}" for k, x in a.items()) + "]"


def _attrs_dec(s: str) -> dict:
    out = {}
    for part in s.split(", "):
        k, sep, x = part.partition("=")
        if not sep:
            raise FormatError(f"bad attribute {part!r}")
        out[k] = x
    return out


_EXTRA_KEYS = ["tag", "id", "class", "attrs", "style", "bind", "on", "size", "draw", "autofocus", "kids"]


def _obj_node_line(n: dict) -> str | None:
    """`> tag#id.class [a=b, c=d] {extras}: text` for a plain element node (children are written below it)."""
    allowed = {"tag", "id", "class", "text", "attrs", "style", "bind", "on", "kids", "size", "draw", "autofocus"}
    if not set(n) <= allowed or any(k in n for k in ("if", "each", "grid", "cell", "then", "else")):
        return None
    head, head_left = _head_enc(n)
    on_is_block = isinstance(n.get("on"), dict) and bool(n["on"]) and all(isinstance(x, list) and x for x in n["on"].values()) \
        and all(_EVENT.match(k) for k in n["on"])
    kids_is_block = isinstance(n.get("kids"), list) and bool(n["kids"])

    def build(attrs_sugar: bool) -> str:
        line = ">" + (" " + head if head else "")
        rest = dict(n)
        for k in ("tag", "id", "class"):
            if k not in head_left:
                rest.pop(k, None)
        rest.pop("text", None)
        if attrs_sugar and "attrs" in rest:
            sugar = _attrs_enc(rest["attrs"])
            if sugar is not None:
                line += " " + sugar
                rest.pop("attrs")
        if on_is_block:
            rest.pop("on")
        if kids_is_block:
            rest.pop("kids")
        extras = {k: rest[k] for k in _EXTRA_KEYS if k in rest}
        if extras:
            line += " " + dumps_min(extras)
        if "text" in n:
            if not _is_str(n["text"]):
                return ""
            line += ": " + _text_enc(n["text"])
        return line

    base = {k: v for k, v in n.items() if not (k == "kids" and kids_is_block) and not (k == "on" and on_is_block)}
    for sugar in (True, False):
        line = build(sugar)
        if not line:
            return None
        if "\n" in line or "\r" in line:
            continue
        try:
            got = _obj_node_parse(line)
        except FormatError:
            continue
        # the line alone must reproduce everything except the pieces written as child lines
        if same(got, base):
            return line
    return None


def _obj_node_parse(line: str) -> dict:
    s = line[1:].lstrip(" ")
    m = _HEAD.match(s)
    out = _head_dec(m.group(0))
    pos = m.end()
    extras: dict = {}

    def seg(ch: str) -> int:  # the start of a segment beginning with `ch`, after one optional space; -1 if absent
        q = pos + 1 if s.startswith(" ", pos) else pos
        return q if s.startswith(ch, q) else -1

    q = seg("[")
    if q >= 0:
        j = s.find("]", q)
        if j < 0:
            raise FormatError("unterminated [attrs]")
        out["attrs"] = _attrs_dec(s[q + 1:j])
        pos = j + 1
    q = seg("{")
    if q >= 0:
        extras, pos = _raw_json(s, q)
        if not isinstance(extras, dict):
            raise FormatError("extras must be an object")
    text_part = None
    if pos < len(s):
        if not s.startswith(": ", pos):
            raise FormatError(f"unexpected {s[pos:pos + 20]!r} in node line")
        text_part = s[pos + 2:]
    if text_part is not None:
        out["text"] = _text_dec(text_part)
    out.update(extras)
    return out


def _var_line(n: dict) -> tuple[str, str] | None:
    """Lines for if / each / grid nodes: (line, kind), verified to read back to the node minus its children."""
    got = _var_line_raw(n)
    if got is None or "\n" in got[0] or "\r" in got[0]:
        return None
    try:
        back = _var_parse(got[0])
    except FormatError:
        return None
    return got if same(back, {k: x for k, x in n.items() if k not in ("then", "else", "kids", "cell")}) else None


def _var_line_raw(n: dict) -> tuple[str, str] | None:
    if "if" in n:
        if set(n) <= {"if", "then", "else"} and _is_str(n["if"]) and n["if"] and isinstance(n.get("then"), list) \
                and ("else" not in n or isinstance(n["else"], list)) and "\n" not in n["if"]:
            return "? " + n["if"], "if"
        return None
    if "each" in n:
        if set(n) <= {"each", "as", "index", "kids"} and _is_str(n["each"]) and _is_str(n.get("as")) \
                and _IDENT.match(n["as"]) and isinstance(n.get("kids"), list) and "\n" not in n["each"] \
                and ("index" not in n or (_is_str(n["index"]) and _IDENT.match(n["index"]))):
            line = f"* {n['each']} as {n['as']}" + (f" index {n['index']}" if "index" in n else "")
            return line, "each"
        return None
    if "grid" in n:
        g = n["grid"]
        if set(n) <= {"grid", "cell", "id", "class", "attrs", "style"} and isinstance(g, list) and len(g) == 2 \
                and all(type(x) is int and x >= 0 for x in g) and isinstance(n.get("cell"), (dict, str)):
            head, left = _head_enc({k: n[k] for k in ("id", "class") if k in n})
            extras = {k: n[k] for k in ("id", "class", "attrs", "style") if k in n and (k in left or k in ("attrs", "style"))}
            line = f"# {g[0]} {g[1]}" + (" " + head if head else "") + (" " + dumps_min(extras) if extras else "")
            return line, "grid"
        return None
    return None


def _var_parse(line: str) -> dict:
    kind = line[0]
    body = line[2:]
    if kind == "?":
        return {"if": body}
    if kind == "*":
        m = re.fullmatch(r"(.*) as ([A-Za-z_][A-Za-z0-9_]*)(?: index ([A-Za-z_][A-Za-z0-9_]*))?", body, re.S)
        if not m:
            raise FormatError(f"bad each line {line[:40]!r}")
        out = {"each": m.group(1), "as": m.group(2)}
        if m.group(3):
            out["index"] = m.group(3)
        return out
    m = re.fullmatch(r"(\d+) (\d+)(.*)", body)
    if not m:
        raise FormatError(f"bad grid line {line[:40]!r}")
    out = {"grid": [int(m.group(1)), int(m.group(2))]}
    rest = m.group(3)
    if rest.startswith(" "):
        rest = rest[1:]
        hm = _HEAD.match(rest)
        out.update(_head_dec(hm.group(0)))
        rest = rest[hm.end():]
        if rest.startswith(" "):
            rest = rest[1:]
    if rest:
        extras, end = _raw_json(rest, 0)
        if rest[end:] or not isinstance(extras, dict):
            raise FormatError(f"bad grid extras in {line[:40]!r}")
        out.update(extras)
    return out


def _emit_node(n: Any, depth: int, out: list[str]) -> None:
    pad = _INDENT * depth
    if isinstance(n, str):
        if re.match(r"[a-z]", n) and n == n.strip() and "\n" not in n and "\r" not in n:
            out.append(pad + n)
        else:
            out.append(pad + "!" + dumps_min(n))
        return
    if isinstance(n, dict):
        line = _obj_node_line(n)
        if line is not None:
            out.append(pad + line)
            # handlers first, then kids (an @ line can never be mistaken for a node line)
            on = n.get("on")
            if isinstance(on, dict) and on and all(isinstance(x, list) and x for x in on.values()) \
                    and all(_EVENT.match(k) for k in on):
                for ev, effs in on.items():
                    out.append(pad + _INDENT + "@" + ev + " -> " + _effects_enc(effs))
            if isinstance(n.get("kids"), list) and n["kids"]:
                for c in n["kids"]:
                    _emit_node(c, depth + 1, out)
            return
        var = _var_line(n)
        if var is not None:
            line, kind = var
            out.append(pad + line)
            if kind == "if":
                for c in n["then"]:
                    _emit_node(c, depth + 1, out)
                if "else" in n:
                    out.append(pad + _INDENT + "|")
                    for c in n["else"]:
                        _emit_node(c, depth + 1, out)
            elif kind == "each":
                for c in n["kids"]:
                    _emit_node(c, depth + 1, out)
            else:
                _emit_node(n["cell"], depth + 1, out)
            return
    out.append(pad + "!" + dumps_min(n))


class _Lines:
    def __init__(self, text: str):
        self.rows: list[tuple[int, str]] = []
        for raw in text.split("\n"):
            if not raw.strip():
                continue
            body = raw.lstrip(" ")
            sp = len(raw) - len(body)
            if sp % len(_INDENT) or "\t" in raw[:sp]:
                raise FormatError(f"bad indentation in {raw[:40]!r}")
            self.rows.append((sp // len(_INDENT), body.rstrip("\r")))
        self.i = 0

    def peek(self):
        return self.rows[self.i] if self.i < len(self.rows) else None


def _parse_nodes(L: _Lines, depth: int) -> list:
    out = []
    while True:
        r = L.peek()
        if r is None or r[0] < depth:
            return out
        if r[0] > depth:
            raise FormatError(f"unexpected indentation before {r[1][:40]!r}")
        out.append(_parse_node(L, depth))


def _parse_node(L: _Lines, depth: int) -> Any:
    _, line = L.rows[L.i]
    L.i += 1
    c = line[0]
    if c == "!":
        v, end = _raw_json(line, 1)
        if line[end:]:
            raise FormatError("trailing text after !json node")
        return v
    if c == ">":
        node = _obj_node_parse(line)
        handlers: dict = {}
        while True:  # handler lines come first
            r = L.peek()
            if r is None or r[0] != depth + 1 or not r[1].startswith("@"):
                break
            L.i += 1
            ev, sep, effs = r[1][1:].partition(" -> ")
            if not sep:
                raise FormatError(f"bad handler line {r[1][:40]!r}")
            handlers[ev] = _effects_dec(effs)
        if handlers:
            node["on"] = handlers
        kids = _parse_nodes(L, depth + 1)
        if kids:
            node["kids"] = kids
        return node
    if c in "?*#" and line[1:2] == " ":
        node = _var_parse(line)
        if c == "?":
            then, els, seen_else = [], [], False
            while True:
                r = L.peek()
                if r is None or r[0] <= depth:
                    break
                if r[0] == depth + 1 and r[1] == "|" and not seen_else:
                    seen_else = True
                    L.i += 1
                    continue
                (els if seen_else else then).append(_parse_node(L, depth + 1))
            node["then"] = then
            if seen_else:
                node["else"] = els
        elif c == "*":
            node["kids"] = _parse_nodes(L, depth + 1)
        else:
            cells = _parse_nodes(L, depth + 1)
            if len(cells) != 1:
                raise FormatError("a grid line needs exactly one cell node")
            node["cell"] = cells[0]
        return node
    return line  # a shorthand string node


def _line_encode(obj: Any) -> str:
    if not isinstance(obj, dict):
        return "!" + dumps_min(obj)
    out: list[str] = []
    for k, v in obj.items():
        if not _KEYNAME.match(k) or k in ("view", "rules") and not isinstance(v, list) or not _simple_key(k):
            out.append("@ " + dumps_min({k: v}))
        elif k == "view" and isinstance(v, list) and v:
            out.append("view:")
            for n in v:
                _emit_node(n, 1, out)
        elif k == "rules" and isinstance(v, list) and v:
            out.append("rules:")
            for r in v:
                out.append(_INDENT + _rule_line(r))
        elif k not in ("view", "rules") and isinstance(v, dict) and v:
            out.append(k + ":")
            for ek, ev in v.items():
                out.append(f"{_INDENT}{_key_enc(ek)}: {_tok_enc(ev)}")
        else:
            out.append(f"{k} {_tok_enc(v)}")
    return "\n".join(out)


def _simple_key(k: str) -> bool:
    # a top-level key written bare must not collide with a reserved marker
    return not k.startswith(("@", "!"))


def _line_decode(text: str) -> Any:
    if text.startswith("!"):
        v, end = _raw_json(text, 1)
        if text[end:].strip():
            raise FormatError("trailing text after !json document")
        return v
    L = _Lines(text)
    out: dict[str, Any] = {}
    while L.peek() is not None:
        depth, line = L.rows[L.i]
        if depth != 0:
            raise FormatError(f"unexpected indentation before {line[:40]!r}")
        L.i += 1
        if line.startswith("@ "):
            v, end = _raw_json(line, 2)
            if not isinstance(v, dict):
                raise FormatError("'@ ' needs a JSON object")
            out.update(v)
        elif line.endswith(":") and _KEYNAME.match(line[:-1]):
            key = line[:-1]
            if key == "view":
                out[key] = _parse_nodes(L, 1)
            elif key == "rules":
                rules = []
                while L.peek() is not None and L.peek()[0] >= 1:
                    d, rl = L.rows[L.i]
                    if d != 1:
                        raise FormatError(f"unexpected indentation before {rl[:40]!r}")
                    L.i += 1
                    rules.append(_rule_parse(rl))
                out[key] = rules
            else:
                m: dict[str, Any] = {}
                while L.peek() is not None and L.peek()[0] >= 1:
                    d, el = L.rows[L.i]
                    if d != 1:
                        raise FormatError(f"unexpected indentation before {el[:40]!r}")
                    L.i += 1
                    k, rest = _key_dec(el)
                    m[k] = _tok_dec(rest)
                out[key] = m
        else:
            key, sep, rest = line.partition(" ")
            if not sep:
                raise FormatError(f"expected 'key value': {line[:40]!r}")
            out[key] = _tok_dec(rest)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Format:
    name: str
    description: str
    encode: Callable[[Any], str]
    decode: Callable[[str], Any]
    model_writable: bool
    writable_note: str
    preserves_key_order: bool

    def roundtrip(self, obj: Any) -> tuple[bool, bool, str]:
        """(structurally equal, key order also equal, the encoded text). A decode error counts as not equal."""
        text = self.encode(obj)
        try:
            back = self.decode(text)
        except (FormatError, ValueError):
            return False, False, text
        return same(obj, back), same_ordered(obj, back), text


FORMATS: dict[str, Format] = {f.name: f for f in [
    Format("json_pretty", "JSON, indent=2: the canonical spec as stored", lambda o: json.dumps(o, indent=2, ensure_ascii=False),
           json.loads, True,
           "Yes, but every indent and newline is a token the model decodes for nothing; here as the reference.", True),
    Format("json_min", "minified JSON: what the primer asks the model to write today", dumps_min, json.loads, True,
           "Yes. This is the current wire format and the one a JSON-schema grammar (Ollama format=) constrains directly.", True),
    Format("json_keys", "minified JSON with a fixed key dictionary by schema position (state->s, rules->r, ...)",
           _keys_encode, _keys_decode, True,
           "Yes under a JSON-schema grammar built from the short keys; without a grammar the dictionary costs prompt "
           "tokens and invites mix-ups (judgement, not measured).", True),
    Format("json_pos", "minified JSON, rules / effects / view nodes as positional arrays", _pos_encode, _pos_decode, False,
           "Doubtful for a 14B: it must count slots and emit null placeholders; ordinary JSON-schema grammars cannot "
           "constrain tuples well (judgement, not measured).", False),
    Format("line", "one rule or view node per line: `on click:#inc -> count = count + 1`", _line_encode, _line_decode, True,
           "Yes with a GBNF grammar (llama.cpp); Ollama's format= takes JSON schema only, so there it is few-shot and "
           "unconstrained, with per-line (not whole-spec) failure (judgement, not measured).", False),
    Format("sexp", "S-expression: (% key value ...) objects, (a b c) lists", _sexp_encode, _sexp_decode, True,
           "Yes with a GBNF grammar, no JSON-schema support; balancing parentheses over 100s of tokens is the risk "
           "(judgement, not measured).", True),
]}


def get_format(name: str) -> Format:
    return FORMATS[name]
