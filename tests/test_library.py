"""Library learning: anti-unification, exact round trip, honest utility accounting, determinism, leave-one-out."""
import os as _os
import pytest as _pytest
if not _os.path.exists(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "symlang/library/data/presets.json")):
    _pytest.skip("private corpus data is not published: regenerate it from a local deterministic-coder checkout (see README, Results so far)", allow_module_level=True)

import json
import os
import random
import shutil
import subprocess

import pytest

from symlang.library import (Abstraction, Cost, Library, anti_unify, apply_library, call, compress,
                             expand, from_json, hole, leave_one_out, match, rewrite, roundtrip_ok,
                             to_json, token_report, transfer)
from symlang.library.learner import def_text, ntok
from symlang.library.report import DATA, FORM_LIST, KERNEL_PRESETS_JS, load_presets, run

J = from_json
PRESETS = load_presets()
CORPUS = {k: J(v) for k, v in PRESETS.items()}


# ----------------------------------------------------------------------------- anti-unification


def test_au_identical_trees_have_no_holes():
    t = J({"a": 1, "b": [1, 2]})
    assert anti_unify(t, t).key == t.key


def test_au_differing_leaf_becomes_hole():
    p = anti_unify(J({"a": 1, "b": 2}), J({"a": 1, "b": 3}))
    assert p.key == '{"a":1,"b":?0}'


def test_au_hole_sharing_follows_the_disagreement_pairs():
    assert anti_unify(J({"x": [1, 1]}), J({"x": [2, 2]})).key == '{"x":[?0,?0]}'
    assert anti_unify(J({"x": [1, 1]}), J({"x": [2, 3]})).key == '{"x":[?0,?1]}'
    # same a-side value, different b-side values: two holes, not one
    assert anti_unify(J([5, 5]), J([6, 7])).key == "[?0,?1]"


def test_au_holes_numbered_by_first_occurrence():
    p = anti_unify(J({"a": 1, "b": {"c": 2, "d": 3}}), J({"a": 9, "b": {"c": 8, "d": 7}}))
    assert p.key == '{"a":?0,"b":{"c":?1,"d":?2}}'


def test_au_subtree_difference_is_one_hole():
    p = anti_unify(J({"k": "v", "body": {"x": 1}}), J({"k": "v", "body": [1, 2, 3]}))
    assert p.key == '{"k":"v","body":?0}'


def test_au_different_keys_or_lengths_generalise_at_the_root():
    assert anti_unify(J({"a": 1}), J({"b": 1})).kind == "H"
    assert anti_unify(J([1, 2]), J([1, 2, 3])).kind == "H"
    assert anti_unify(J({"a": 1, "b": 2}), J({"b": 2, "a": 1})).kind == "H"  # key order is part of the shape


def test_au_pattern_matches_both_inputs_and_substitution_reproduces_them():
    a = J({"tag": "button", "id": "inc", "text": "+", "attrs": {"type": "button"}})
    b = J({"tag": "button", "id": "dec", "text": "-", "attrs": {"type": "button"}})
    p = anti_unify(a, b)
    for t in (a, b):
        bind = {}
        assert match(p, t, bind)
        assert rewrite(t, p, "g", 2).key == call("g", [bind[0], bind[1]]).key
    lib = Library({"g": Abstraction("g", 2, p, p)})
    assert expand(rewrite(a, p, "g", 2), lib).key == a.key


def test_au_distinguishes_bool_from_int_and_float_from_int():
    assert anti_unify(J([True]), J([1])).key == "[?0]"
    assert anti_unify(J([1]), J([1.0])).key == "[?0]"
    assert J([True]) != J([1])


def test_match_repeated_hole_needs_equal_bindings():
    p = anti_unify(J([5, 5]), J([6, 6]))
    assert p.key == "[?0,?0]"
    assert match(p, J([9, 9]), {})
    assert not match(p, J([9, 8]), {})


def test_match_never_binds_through_a_constant_mismatch():
    p = anti_unify(J({"a": 1, "b": 2}), J({"a": 1, "b": 3}))
    assert not match(p, J({"a": 2, "b": 2}), {})


def test_rewrite_finds_nested_uses_inside_arguments():
    p = anti_unify(J({"w": [1, 9]}), J({"w": [2, 9]}))        # {"w":[?0,9]}
    t = J({"w": [{"w": [3, 9]}, 9]})
    r = rewrite(t, p, "g", 1)
    assert r.key == '@g(@g(3))'
    lib = Library({"g": Abstraction("g", 1, p, p)})
    assert expand(r, lib).key == t.key


# ----------------------------------------------------------------------------- round trip


@pytest.mark.parametrize("objective", ["nodes", "tokens"])
def test_roundtrip_six_presets_exact(objective):
    res = compress(CORPUS, objective)
    assert len(res.library) > 0
    for k, orig in res.originals.items():
        back = expand(res.rewritten[k], res.library)
        assert back.key == orig.key
        assert json.dumps(to_json(back), ensure_ascii=False) == json.dumps(PRESETS[k], ensure_ascii=False)
        assert to_json(back) == PRESETS[k]
    assert roundtrip_ok(res)


def _rand_leaf(r):
    return r.choice([0, 1, 2, -3, 1.0, 2.5, True, False, None, "", "a", "b", "{x}", "click:#a", "é", "q\"uote"])


def _rand_tree(r, depth, vocab):
    if depth == 0 or r.random() < 0.25:
        return _rand_leaf(r)
    if r.random() < 0.5:
        keys = r.choice(vocab)
        return {k: _rand_tree(r, depth - 1, vocab) for k in keys}
    return [_rand_tree(r, depth - 1, vocab) for _ in range(r.randint(0, 4))]


def _rand_corpus(seed):
    r = random.Random(seed)
    vocab = [("a",), ("a", "b"), ("tag", "id", "text"), ("on", "do"), ("x", "y", "z")]
    motifs = [_rand_tree(r, 3, vocab) for _ in range(3)]
    corpus = {}
    for i in range(r.randint(2, 4)):
        items = []
        for _ in range(r.randint(4, 9)):
            m = r.choice(motifs)
            items.append(_mutate(m, r, vocab))
        corpus[f"s{i}"] = {"v": 1, "items": items, "tail": _rand_tree(r, 3, vocab)}
    return corpus


def _mutate(t, r, vocab):
    if isinstance(t, dict):
        return {k: (_rand_tree(r, 2, vocab) if r.random() < 0.15 else _mutate(v, r, vocab)) for k, v in t.items()}
    if isinstance(t, list):
        return [_mutate(v, r, vocab) for v in t]
    return _rand_leaf(r) if r.random() < 0.3 else t


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("objective", ["nodes", "tokens"])
def test_roundtrip_random_trees_property(seed, objective):
    corpus = _rand_corpus(seed)
    res = compress(corpus, objective)
    res.library.validate()
    for k, v in corpus.items():
        back = to_json(expand(res.rewritten[k], res.library))
        assert json.dumps(back, ensure_ascii=False) == json.dumps(v, ensure_ascii=False)
    # and a held-out random tree compressed with that library
    held = _rand_corpus(seed + 1000)["s0"]
    comp = apply_library(res.library, J(held))
    assert json.dumps(to_json(expand(comp, res.library)), ensure_ascii=False) == json.dumps(held, ensure_ascii=False)


def test_random_corpora_actually_exercise_the_learner():
    assert sum(len(compress(_rand_corpus(s)).library) for s in range(10)) > 0


# ----------------------------------------------------------------------------- utility accounting


@pytest.mark.parametrize("objective", ["nodes", "tokens"])
def test_claimed_utilities_telescope_to_the_real_before_after(objective):
    """Every accepted step's utility is positive, and their sum equals (size of the original corpus)
    minus (size of the rewritten corpus + size of the library), recomputed from scratch."""
    res = compress(CORPUS, objective)
    cost = Cost(objective)
    before = sum(cost.size(t) for t in res.originals.values())
    after = sum(cost.size(t) for t in res.rewritten.values()) + res.library.size(cost)
    assert all(s.utility > 0 for s in res.steps)
    assert sum(s.utility for s in res.steps) == before - after
    assert before - after > 0


def test_token_report_matches_a_fresh_tiktoken_count():
    res = compress(CORPUS, "nodes")
    tr = token_report(res)
    for k, t in res.rewritten.items():
        assert tr["comp"][k] == ntok(t.key)
    assert tr["library"] == sum(ntok(def_text(a.name, a.arity, a.body)) for a in res.library)
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    for k, v in PRESETS.items():
        assert tr["orig"][k] == len(enc.encode(json.dumps(v, separators=(",", ":"), ensure_ascii=False)))


def test_no_saving_is_claimed_when_nothing_repeats():
    res = compress({"a": {"p": 1, "q": 2}, "b": [1, 2, 3, 4], "c": "x"})
    assert len(res.library) == 0 and res.steps == []
    assert res.rewritten["a"].key == res.originals["a"].key


def test_a_definition_must_pay_for_itself():
    # one use of a 3-atom pattern cannot beat its definition
    res = compress({"a": {"k": 1, "m": 2, "n": 3}})
    assert len(res.library) == 0


def test_every_call_in_the_corpus_expands_to_an_instance_of_its_pattern():
    from symlang.library.tree import iter_sites
    res = compress(CORPUS)
    n_calls = 0
    for t in res.rewritten.values():
        for _, n in iter_sites(t):
            if n.kind == "C":
                n_calls += 1
                a = res.library.defs[n.val]
                assert len(n.kids) == a.arity
                inst = expand(n, res.library)
                assert match(expand(a.pattern, res.library), inst, {})
    assert n_calls > 50


def test_library_validate_rejects_cycles_and_bad_arity():
    p = J({"a": 1})
    lib = Library({"f": Abstraction("f", 0, p, call("g", [])), "g": Abstraction("g", 0, p, call("f", []))})
    with pytest.raises(ValueError):
        lib.validate()
    with pytest.raises(ValueError):
        Library({"f": Abstraction("f", 1, p, call("f2", [hole(0)]))}).validate()


def test_hierarchical_expand_inlines_nested_calls():
    from symlang.library import mk_dict
    inner = J({"k": 1})
    g = Abstraction("g", 0, inner, inner)
    body = mk_dict(["top", "arg"], [call("g", []), hole(0)])   # f(?0) = {"top": g(), "arg": ?0}
    lib = Library({"f": Abstraction("f", 1, body, body), "g": g})
    lib.validate()
    out = expand(call("f", [J(5)]), lib)
    assert to_json(out) == {"top": {"k": 1}, "arg": 5}


# ----------------------------------------------------------------------------- determinism


def _fingerprint(res):
    return "\n".join(def_text(a.name, a.arity, a.body) + "|" + a.pattern.key for a in res.library) + "\n" + \
        "\n".join(f"{k}={v.key}" for k, v in sorted(res.rewritten.items()))


@pytest.mark.parametrize("objective", ["nodes", "tokens"])
def test_deterministic_byte_identical(objective):
    a = _fingerprint(compress(CORPUS, objective))
    b = _fingerprint(compress({k: from_json(v) for k, v in PRESETS.items()}, objective))
    assert a == b
    # insertion order of the corpus must not matter either
    shuffled = dict(reversed(list(CORPUS.items())))
    assert _fingerprint(compress(shuffled, objective)) == a


def test_deterministic_report_text():
    assert run("nodes") == run("nodes")


# ----------------------------------------------------------------------------- leave-one-out


def test_leave_one_out_harness_runs_and_is_consistent():
    out = leave_one_out(CORPUS, "nodes")
    assert sorted(out["rows"]) == sorted(PRESETS)
    for k, r in out["rows"].items():
        assert r["loo_saving"] >= 0          # a held-out rewrite never grows a tree under the node cost
        assert r["in_saving"] >= 0
        assert r["orig"] == CORPUS[k].nodes
        assert r["abstractions_used"] <= r["abstractions_learned"]
        assert r["loo_comp"] == r["orig"] - r["loo_saving"]
    # in-sample library has seen the spec, so on the whole it saves at least as much as the LOO one
    assert sum(r["in_saving"] for r in out["rows"].values()) >= sum(r["loo_saving"] for r in out["rows"].values())


def test_leave_one_out_library_never_saw_the_held_out_spec():
    held = "counter"
    train = {k: v for k, v in CORPUS.items() if k != held}
    res = compress(train)
    assert held not in res.originals and held not in res.rewritten
    comp = apply_library(res.library, CORPUS[held])
    assert expand(comp, res.library).key == CORPUS[held].key


def test_transfer_from_form_list_presets_runs():
    out = transfer(CORPUS, FORM_LIST, "nodes")
    assert {k for k, r in out["rows"].items() if r["train"]} == set(FORM_LIST)
    assert all(r["saving"] >= 0 for r in out["rows"].values())


def test_report_runs_for_both_objectives():
    for obj in ("nodes", "tokens"):
        text = run(obj)
        assert "leave-one-out" in text and "round trip expand(compress(x)) == x: True" in text


# ----------------------------------------------------------------------------- data


def test_presets_json_has_the_six_kernel_specs():
    assert sorted(PRESETS) == ["bmi-calculator", "counter", "landing-page", "snake", "tic-tac-toe", "todo"]
    for v in PRESETS.values():
        assert v["v"] == 1 and "view" in v and "state" in v


@pytest.mark.skipif(not (shutil.which("node") and os.path.exists(KERNEL_PRESETS_JS)), reason="needs node and the deterministic-coder repo")
def test_presets_json_matches_a_fresh_dump_from_the_source_repo():
    here = os.path.dirname(DATA)
    out = subprocess.run(["node", os.path.join(here, "..", "dump_presets.js"), KERNEL_PRESETS_JS],
                         check=True, capture_output=True, text=True).stdout
    assert json.loads(out) == PRESETS
