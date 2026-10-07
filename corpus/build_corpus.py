"""Build the real-artifact corpus for the wire-format study.

Gathers three kinds of artifact from the Deterministic Coder repo (READ ONLY) into corpus/data/:

  prompts.json        the 24 held-out natural-language prompts (test/eval/heldout/*.json, field `prompt`)
  presets.json        the six Kernel preset specs, canonical long form, exported by evaluating the repo's browser
                      scripts in a Node vm context (corpus/export_kernel.js), the same way test/kernel_presets.test.js does
  replies.json        every recorded raw model reply of test/fixtures/kernel_replay/*.json (what the local model
                      actually wrote), with the replay metadata that matters here
  kernel_prompt.json  the Kernel primer and response schema (per-call prompt overhead), from the same loader
  manifest.json       provenance (source path, sha256, git commit of the source repo) and sizes

Usage:  ./.venv/bin/python -m corpus.build_corpus [--src /path/to/deterministic-coder] [--out corpus/data]
Deterministic: the same sources give byte-identical output (no timestamps).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
def _default_src():
    """Private Deterministic Coder checkout: $DETERMINISTIC_CODER_REPO, else a sibling directory (the data is not published)."""
    import os
    for c in (os.environ.get("DETERMINISTIC_CODER_REPO", ""), HERE.parents[1] / "deterministic-coder", HERE.parents[2] / "deterministic-coder"):
        if c and Path(c).is_dir():
            return Path(c)
    return HERE.parents[1] / "deterministic-coder"


DEFAULT_SRC = _default_src()
DEFAULT_OUT = HERE / "data"
EXPORTER = HERE / "export_kernel.js"
MAX_CORPUS_BYTES = 1_000_000

REPLY_KEEP = ["id", "group", "run", "ok", "stage", "firstStage", "repaired", "cls", "firstCls", "promptTokens",
              "outputTokens", "sec", "truncated", "turns", "oraclePass"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(obj, path: Path) -> int:
    text = json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def git_head(src: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty(src: Path, paths: list[str]) -> list[str]:
    """Which of `paths` (relative to src) are modified or untracked in the source repo right now."""
    try:
        out = subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--"] + paths, capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return []
    return sorted(line[3:] for line in out.splitlines() if line.strip())


def export_kernel(src: Path) -> dict:
    """Run the Node loader. Returns {presets, primer, schema}."""
    backend = src / "ui" / "public" / "backend"
    proc = subprocess.run(["node", str(EXPORTER), str(backend)], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"node export failed: {proc.stderr.strip()[:500]}")
    return json.loads(proc.stdout)


def load_prompts(src: Path):
    d = src / "test" / "eval" / "heldout"
    out, srcs = [], []
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        out.append({"id": j["id"], "group": j.get("group"), "domain": j.get("domain"),
                    "expectArchetype": j.get("expectArchetype"), "difficulty": j.get("difficulty"),
                    "prompt": j["prompt"]})
        srcs.append(f)
    return out, srcs


def load_replies(src: Path):
    d = src / "test" / "fixtures" / "kernel_replay"
    out, srcs = [], []
    for f in sorted(d.glob("*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        for r in rows:
            raw = r.get("raw") or ""
            try:
                parsed = json.loads(raw) if raw else None
                parses = isinstance(parsed, dict)
            except ValueError:
                parses = False
            e = {"source": f.stem}
            e.update({k: r.get(k) for k in REPLY_KEEP})
            e["parses"] = parses
            e["raw"] = raw
            out.append(e)
        srcs.append(f)
    return out, srcs


def build(src: Path = DEFAULT_SRC, out: Path = DEFAULT_OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    prompts, prompt_files = load_prompts(src)
    replies, reply_files = load_replies(src)
    k = export_kernel(src)
    files = {
        "prompts.json": prompts,
        "presets.json": k["presets"],
        "replies.json": replies,
        "kernel_prompt.json": {"primer": k["primer"], "schema": k["schema"]},
    }
    sizes = {name: dump(obj, out / name) for name, obj in files.items()}
    backend = src / "ui" / "public" / "backend"
    sources = [{"path": str(p.relative_to(src)), "sha256": sha256(p)} for p in
               prompt_files + reply_files + [backend / n for n in
                                              ("kernel_expr.js", "kernel_core.js", "kernel_presets.js", "kernel_model.js")]]
    by_source = {}
    for r in replies:
        s = by_source.setdefault(r["source"], {"replies": 0, "parse_as_json_object": 0, "empty": 0, "truncated": 0})
        s["replies"] += 1
        s["parse_as_json_object"] += 1 if r["parses"] else 0
        s["empty"] += 1 if not r["raw"] else 0
        s["truncated"] += 1 if r.get("truncated") else 0
    manifest = {
        "description": "Real artifacts of the Deterministic Coder Kernel, for measuring wire formats in model tokens.",
        "source_repo": str(src),
        "source_git_head": git_head(src),
        "source_dirty": git_dirty(src, [x["path"] for x in sources]),
        "note": "The source repo is edited while this corpus is built; source_dirty lists the used files that differ "
                "from source_git_head, and sha256 pins the exact bytes used.",
        "counts": {"prompts": len(prompts), "presets": len(k["presets"]), "replies": len(replies),
                   "replies_parse_as_json_object": sum(1 for r in replies if r["parses"])},
        "replies_by_source": by_source,
        "preset_ids": list(k["presets"]),
        "kernel_prompt": {"primer_chars": len(k["primer"]), "schema_chars": len(json.dumps(k["schema"], separators=(",", ":")))},
        "file_bytes": sizes,
        "sources": sources,
    }
    sizes["manifest.json"] = dump(manifest, out / "manifest.json")
    total = sum(sizes.values())
    manifest["file_bytes"] = sizes
    manifest["total_bytes"] = total
    dump(manifest, out / "manifest.json")
    if total > MAX_CORPUS_BYTES:
        raise RuntimeError(f"corpus is {total} bytes, over the {MAX_CORPUS_BYTES} budget")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    m = build(a.src, a.out)
    print(json.dumps({"counts": m["counts"], "total_bytes": m["total_bytes"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
