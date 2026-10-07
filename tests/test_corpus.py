"""The corpus of real Kernel artifacts: committed snapshot invariants, and the builder against the live source repo."""
import os as _os
import pytest as _pytest
if not _os.path.exists(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "corpus", "data", "manifest.json")):
    _pytest.skip("private corpus data is not published: regenerate it from a local deterministic-coder checkout (python -m corpus.build_corpus)", allow_module_level=True)

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from corpus import build_corpus as bc

DATA = Path(bc.DEFAULT_OUT)
SRC = bc.DEFAULT_SRC
needs_src = pytest.mark.skipif(not (SRC / "ui" / "public" / "backend" / "kernel_presets.js").exists() or not shutil.which("node"),
                               reason="Deterministic Coder repo or node not available")


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_snapshot_counts():
    m = load("manifest.json")
    assert m["counts"] == {"prompts": 24, "presets": 6, "replies": 48, "replies_parse_as_json_object": 31}
    assert len(load("prompts.json")) == 24
    presets = load("presets.json")
    assert sorted(presets) == ["bmi-calculator", "counter", "landing-page", "snake", "tic-tac-toe", "todo"]
    assert len(load("replies.json")) == 48
    assert m["preset_ids"] == list(presets)


def test_snapshot_is_small_and_pinned():
    m = load("manifest.json")
    total = sum(p.stat().st_size for p in DATA.glob("*.json"))
    assert total < 1_000_000
    assert m["total_bytes"] <= total  # the manifest records its own size before the final rewrite
    assert all(len(s["sha256"]) == 64 for s in m["sources"])
    assert any(s["path"].endswith("kernel_model.js") for s in m["sources"])


def test_prompts_are_real_prompts():
    prompts = load("prompts.json")
    assert len({p["id"] for p in prompts}) == 24
    assert all(isinstance(p["prompt"], str) and len(p["prompt"]) > 10 for p in prompts)


def test_replies_flags_are_honest():
    for r in load("replies.json"):
        assert r["parses"] == (bool(r["raw"]) and isinstance(_try(r["raw"]), dict))
    by = load("manifest.json")["replies_by_source"]
    assert by["14b_all24_v2"]["replies"] == 24
    assert sum(v["parse_as_json_object"] for v in by.values()) == 31


def _try(raw):
    try:
        return json.loads(raw)
    except ValueError:
        return None


def test_kernel_prompt_present():
    kp = load("kernel_prompt.json")
    assert kp["primer"].startswith("Write ONE JSON app spec")
    assert "Example 1: {" in kp["primer"] and "Example 2: {" in kp["primer"]
    assert kp["schema"]["required"] == ["title", "state", "view", "rules"]


@needs_src
def test_builder_is_deterministic_and_matches_the_source(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ma = bc.build(SRC, a)
    mb = bc.build(SRC, b)
    for name in ("prompts.json", "presets.json", "replies.json", "kernel_prompt.json", "manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert ma["counts"]["prompts"] == 24 and ma["counts"]["presets"] == 6
    assert ma["counts"]["replies"] >= 48
    assert ma["total_bytes"] < 1_000_000
    assert ma == mb


@needs_src
def test_node_exporter_returns_the_six_presets_and_the_primer():
    k = bc.export_kernel(SRC)
    assert sorted(k["presets"]) == ["bmi-calculator", "counter", "landing-page", "snake", "tic-tac-toe", "todo"]
    assert k["primer"].startswith("Write ONE JSON app spec")
    assert isinstance(k["schema"], dict)
