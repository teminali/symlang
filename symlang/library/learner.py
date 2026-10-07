"""Compression-driven library learning (minimum description length), Stitch-style greedy loop.

    total description length  =  L(library) + sum_specs L(spec | library)

Pieces:
  * anti_unify(a, b)   least general generalisation of two trees -> pattern with numbered holes.
  * match / rewrite    top-down, non-overlapping rewriting of a tree with one abstraction.
  * compress(corpus)   greedy loop: pick the abstraction with the best utility, rewrite the corpus
                       AND earlier definitions with it, repeat while utility > 0.
  * expand(tree, lib)  exact inverse. ``expand(compress(x)) == x`` is the round-trip guarantee.
  * apply_library      compress a NEW tree with an already learned library (leave-one-out).

Utility is *estimated* cheaply to rank candidates, but every abstraction that is accepted is
re-measured by really rewriting the corpus: the claimed saving is always before-minus-after of real
trees, never the estimate (see ``Step.utility``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from .tree import (Node, call, from_json, hole, iter_sites, leaf, mk_dict, mk_list, to_json,
                   with_kids)

# --------------------------------------------------------------------------- cost model


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=None)
def ntok(text: str) -> int:
    """cl100k_base token count of a text."""
    return len(_encoder().encode(text, disallowed_special=()))


def def_text(name: str, arity: int, body: Node) -> str:
    """How a definition is written down: ``@f3(?0,?1)=<body>``. Token cost of the library is the
    token count of these lines."""
    return f"@{name}(" + ",".join(f"?{i}" for i in range(arity)) + ")=" + body.key


class Cost:
    """``nodes``: atoms (leaf=1, dict=1+#keys, list=1, call=1, children added).
    ``tokens``: cl100k tokens of the minified text of the subtree (calls written ``@f3(a,b)``)."""

    def __init__(self, objective: str = "nodes"):
        if objective not in ("nodes", "tokens"):
            raise ValueError(objective)
        self.objective = objective

    def size(self, n: Node) -> int:
        return n.nodes if self.objective == "nodes" else ntok(n.key)

    def call_size(self, name: str, args) -> int:
        if self.objective == "nodes":
            return 1 + sum(a.nodes for a in args)
        return ntok(call(name, args).key)

    def def_size(self, name: str, arity: int, body: Node) -> int:
        if self.objective == "nodes":
            return 1 + body.nodes
        return ntok(def_text(name, arity, body))


# --------------------------------------------------------------------------- anti-unification


def anti_unify(a: Node, b: Node) -> Node:
    """Least general generalisation: the most specific pattern that matches both ``a`` and ``b``.

    Same shape (dict with identical key sequence, list of equal length, call of the same
    abstraction and arity) -> recurse. Anything else that differs -> a hole. A disagreement pair
    seen twice (same subtree of ``a`` against the same subtree of ``b``) reuses ONE hole, so
    ``f(x, x)`` vs ``f(y, y)`` generalises to ``f(?0, ?0)``, not ``f(?0, ?1)``. Holes are numbered
    by first occurrence in a left-to-right preorder walk, so equal patterns have equal keys.
    """
    table: dict[tuple[str, str], Node] = {}

    def go(x: Node, y: Node) -> Node:
        if x.nh == 0 and y.nh == 0 and x.key == y.key:
            return x
        if x.kind == y.kind and x.kind in "DSC" and x.shape == y.shape:
            kids = [go(p, q) for p, q in zip(x.kids, y.kids)]
            if x.kind == "D":
                return mk_dict(x.keys, kids)
            if x.kind == "S":
                return mk_list(kids)
            return call(x.val, kids)
        k = (x.key, y.key)
        h = table.get(k)
        if h is None:
            h = table[k] = hole(len(table))
        return h

    return go(a, b)


def arity_of(p: Node) -> int:
    hs = [-1]

    def go(n):
        if n.kind == "H":
            hs.append(n.val)
        for c in n.kids:
            go(c)
    go(p)
    return max(hs) + 1


def const_nodes(p: Node) -> int:
    """Atoms of the pattern that are not holes (what a use of the abstraction saves writing)."""
    return p.nodes - p.nh


def match(p: Node, n: Node, b: dict) -> bool:
    """Does pattern ``p`` match tree ``n``? Fills ``b`` with hole -> subtree. A hole used twice must
    bind equal subtrees. Constant parts must be identical, so a Hole already inside ``n`` (a
    parameter of an enclosing definition) only ever matches a pattern hole."""
    if p.kind == "H":
        prev = b.get(p.val)
        if prev is None:
            b[p.val] = n
            return True
        return prev.key == n.key
    if p.nh == 0:
        return p.key == n.key
    if p.kind != n.kind or p.shape != n.shape:
        return False
    for pk, nk in zip(p.kids, n.kids):
        if not match(pk, nk, b):
            return False
    return True


def rewrite(n: Node, p: Node, name: str, arity: int) -> Node:
    """Replace matches of ``p`` by ``@name(args)``, top-down: a match consumes its constant part
    (matches inside it are impossible, they are part of the body) but the arguments are rewritten
    recursively, so nested uses are found."""
    if n.kind in "DSC" and n.shape == p.shape:
        b: dict = {}
        if match(p, n, b):
            return call(name, [rewrite(b[i], p, name, arity) for i in range(arity)])
    if not n.kids:
        return n
    return with_kids(n, [rewrite(c, p, name, arity) for c in n.kids])


# --------------------------------------------------------------------------- library


@dataclass(frozen=True)
class Abstraction:
    name: str
    arity: int
    pattern: Node   # as learned (only calls to EARLIER abstractions): what apply_library matches
    body: Node      # final definition, rewritten with LATER abstractions: what expand() uses


@dataclass
class Library:
    defs: dict[str, Abstraction] = field(default_factory=dict)

    def __len__(self):
        return len(self.defs)

    def __iter__(self):
        return iter(self.defs.values())

    def size(self, cost: Cost) -> int:
        return sum(cost.def_size(a.name, a.arity, a.body) for a in self)

    def validate(self) -> None:
        """Arity matches the holes, every call is to a defined abstraction with the right arity,
        and the reference graph is acyclic (so expand terminates)."""
        for a in self:
            holes = set()

            def go(n):
                if n.kind == "H":
                    holes.add(n.val)
                if n.kind == "C":
                    t = self.defs.get(n.val)
                    if t is None or t.arity != len(n.kids):
                        raise ValueError(f"{a.name}: bad call {n.key[:50]}")
                for c in n.kids:
                    go(c)
            go(a.body)
            if holes - set(range(a.arity)):
                raise ValueError(f"{a.name}: hole outside arity")
        state: dict[str, int] = {}

        def visit(name):
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                raise ValueError(f"cycle through {name}")
            state[name] = 1

            def refs(n):
                if n.kind == "C":
                    visit(n.val)
                for c in n.kids:
                    refs(c)
            refs(self.defs[name].body)
            state[name] = 2
        for nm in self.defs:
            visit(nm)


def _subst(body: Node, args) -> Node:
    if body.kind == "H":
        return args[body.val]
    if not body.kids:
        return body
    return with_kids(body, [_subst(c, args) for c in body.kids])


def expand(n: Node, lib: Library) -> Node:
    """Inline every call, recursively. Exact inverse of rewriting."""
    if n.kind == "C":
        a = lib.defs[n.val]
        args = [expand(c, lib) for c in n.kids]
        return expand(_subst(a.body, args), lib)
    if not n.kids:
        return n
    return with_kids(n, [expand(c, lib) for c in n.kids])


def apply_library(lib: Library, tree: Node) -> Node:
    """Compress a tree with a fixed library, abstractions applied in the order they were learned."""
    for a in lib:
        tree = rewrite(tree, a.pattern, a.name, a.arity)
    return tree


# --------------------------------------------------------------------------- discovery


class _Pool:
    """Candidate patterns from pairwise anti-unification, memoised across greedy iterations."""

    def __init__(self, min_const: int, max_arity: int, min_site: int, max_group: int):
        self.min_const, self.max_arity, self.min_site, self.max_group = min_const, max_arity, min_site, max_group
        self.memo: dict[tuple[str, str], Node | None] = {}

    def valid(self, p: Node | None) -> bool:
        return (p is not None and p.kind in "DSC" and const_nodes(p) >= self.min_const
                and arity_of(p) <= self.max_arity)

    def pair(self, a: Node, b: Node) -> Node | None:
        k = (a.key, b.key)
        if k not in self.memo:
            p = anti_unify(a, b)
            self.memo[k] = p if self.valid(p) else None
        return self.memo[k]

    def discover(self, corpus: list[tuple[str, Node]]) -> dict[str, Node]:
        groups: dict[tuple, dict[str, Node]] = {}
        seen_count: dict[str, int] = {}
        for _, root in corpus:
            for _, n in iter_sites(root):
                if n.nodes >= self.min_site and n.nh == 0:
                    groups.setdefault(n.shape, {})[n.key] = n
                    seen_count[n.key] = seen_count.get(n.key, 0) + 1
        cands: dict[str, Node] = {}
        for shape in sorted(groups, key=repr):
            g = groups[shape]
            nodes = [g[k] for k in sorted(g)]
            if len(nodes) > self.max_group:  # keep the biggest, deterministic
                nodes = sorted(nodes, key=lambda x: (-x.nodes, x.key))[: self.max_group]
            for n in nodes:
                if seen_count[n.key] >= 2 and self.valid(n):
                    cands[n.key] = n  # an exactly repeated subtree: a zero-argument abstraction
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    p = self.pair(nodes[i], nodes[j])
                    if p is not None:
                        cands[p.key] = p
        return cands

    def refine(self, pats: list[Node], corpus: list[tuple[str, Node]]) -> dict[str, Node]:
        """lgg(pattern, subtree): lets a pattern learned from two instances widen to cover a third."""
        by_shape: dict[tuple, dict[str, Node]] = {}
        for _, root in corpus:
            for _, n in iter_sites(root):
                if n.nodes >= self.min_site and n.nh == 0:
                    by_shape.setdefault(n.shape, {})[n.key] = n
        out: dict[str, Node] = {}
        for p in pats:
            for k in sorted(by_shape.get(p.shape, {})):
                n = by_shape[p.shape][k]
                b: dict = {}
                if match(p, n, b):
                    continue
                q = anti_unify(p, n)
                if self.valid(q):
                    out[q.key] = q
        return out


def _const_paths(p: Node, prefix=()) -> list[tuple]:
    """Relative paths of the dict/list/call nodes a match of ``p`` consumes (everything not under a hole)."""
    if p.kind == "H":
        return []
    out = [prefix] if p.kind in "DSC" else []
    for i, c in enumerate(p.kids):
        out.extend(_const_paths(c, prefix + (i,)))
    return out


def _estimate(p: Node, name: str, arity: int, sites: dict, cost: Cost) -> tuple[int, int]:
    """Cheap utility estimate on the current corpus: sum of (size(match) - size(call)) over the
    non-overlapping matches, minus the definition. Used to RANK; never reported as a result."""
    consumed: set = set()
    rel = None
    total = uses = 0
    for tid, path, n in sites.get(p.shape, ()):
        if (tid, path) in consumed:
            continue
        b: dict = {}
        if not match(p, n, b):
            continue
        if rel is None:
            rel = _const_paths(p)
        for r in rel:
            consumed.add((tid, path + r))
        total += cost.size(n) - cost.call_size(name, [b[i] for i in range(arity)])
        uses += 1
    return total - cost.def_size(name, arity, p), uses


# --------------------------------------------------------------------------- the greedy loop


@dataclass
class Step:
    name: str
    arity: int
    pattern: Node
    estimate: int
    utility: int   # MEASURED: total size before minus total size after (corpus + library), same cost model
    uses: int      # calls to it right after it was added


@dataclass
class Result:
    library: Library
    rewritten: dict[str, Node]
    originals: dict[str, Node]
    steps: list[Step]
    objective: str

    def sizes(self, cost: Cost | None = None) -> dict:
        cost = cost or Cost(self.objective)
        orig = {k: cost.size(v) for k, v in self.originals.items()}
        comp = {k: cost.size(v) for k, v in self.rewritten.items()}
        return {"orig": orig, "comp": comp, "library": self.library.size(cost)}


def compress(corpus: dict, objective: str = "nodes", max_abstractions: int = 60, min_const: int = 3,
             max_arity: int = 8, min_site: int = 4, max_group: int = 160, refine: bool = True,
             top_refine: int = 12, verify_top: int = 6) -> Result:
    """Learn a library from ``corpus`` ({id: JSON value or Node}). Deterministic: iteration order is
    sorted, ties break on the pattern text."""
    cost = Cost(objective)
    originals = {k: (v if isinstance(v, Node) else from_json(v)) for k, v in sorted(corpus.items())}
    trees = dict(originals)
    defs: dict[str, Abstraction] = {}
    steps: list[Step] = []
    pool = _Pool(min_const, max_arity, min_site, max_group)

    def total() -> int:
        return (sum(cost.size(t) for t in trees.values())
                + sum(cost.def_size(a.name, a.arity, a.body) for a in defs.values()))

    while len(defs) < max_abstractions:
        name = f"f{len(defs)}"
        corpus_list = sorted(trees.items())
        # sites over the corpus AND earlier definitions: later abstractions may shrink them too
        site_trees = corpus_list + [("def:" + a.name, a.body) for a in defs.values()]
        sites: dict[tuple, list] = {}
        for tid, root in site_trees:
            for path, n in iter_sites(root):
                sites.setdefault(n.shape, []).append((tid, path, n))
        cands = pool.discover(corpus_list)
        scored = []
        for k in sorted(cands):
            p = cands[k]
            est, uses = _estimate(p, name, arity_of(p), sites, cost)
            scored.append((est, k, p))
        if refine and scored:
            scored.sort(key=lambda t: (-t[0], t[1]))
            extra = pool.refine([p for _, _, p in scored[:top_refine]], corpus_list)
            for k in sorted(extra):
                if k not in cands:
                    p = extra[k]
                    est, _ = _estimate(p, name, arity_of(p), sites, cost)
                    scored.append((est, k, p))
        scored.sort(key=lambda t: (-t[0], t[1]))
        before = total()
        accepted = None
        for est, k, p in scored[:verify_top]:
            if est <= 0:
                break
            ar = arity_of(p)
            new_trees = {t: rewrite(v, p, name, ar) for t, v in trees.items()}
            new_defs = {nm: Abstraction(nm, a.arity, a.pattern, rewrite(a.body, p, name, ar))
                        for nm, a in defs.items()}
            new_defs[name] = Abstraction(name, ar, p, p)
            old_trees, old_defs = trees, defs
            trees, defs = new_trees, new_defs
            real = before - total()
            if real > 0:
                uses = sum(1 for t in trees.values() for _, n in iter_sites(t) if n.kind == "C" and n.val == name)
                accepted = Step(name, ar, p, est, real, uses)
                break
            trees, defs = old_trees, old_defs  # estimate was wrong: reject, try the next
        if accepted is None:
            break
        steps.append(accepted)
    lib = Library(dict(defs))
    lib.validate()
    return Result(lib, trees, originals, steps, objective)


# --------------------------------------------------------------------------- evaluation


def roundtrip_ok(res: Result) -> bool:
    """expand(rewritten) == original, exactly, for every spec (tree key AND plain JSON value)."""
    import json
    for k, orig in res.originals.items():
        back = expand(res.rewritten[k], res.library)
        if back.key != orig.key:
            return False
        if json.dumps(to_json(back), ensure_ascii=False) != json.dumps(to_json(orig), ensure_ascii=False):
            return False
    return True


def token_report(res: Result) -> dict:
    """Real cl100k tokens of the minified text (calls written ``@f3(a,b)``), per spec and total."""
    orig = {k: ntok(v.key) for k, v in res.originals.items()}
    comp = {k: ntok(v.key) for k, v in res.rewritten.items()}
    lib = sum(ntok(def_text(a.name, a.arity, a.body)) for a in res.library)
    return {"orig": orig, "comp": comp, "library": lib}


def leave_one_out(corpus: dict, objective: str = "nodes", **kw) -> dict:
    """Learn on all-but-one, compress the held-out spec with that library, compare with the saving
    it gets from the library learned on all of them (in-sample). Savings are gross of library cost
    in both cases (the library is shared by every spec that will ever use it)."""
    cost = Cost(objective)
    full = compress(corpus, objective, **kw)
    rows = {}
    for held in sorted(corpus):
        train = {k: v for k, v in corpus.items() if k != held}
        res = compress(train, objective, **kw)
        orig = full.originals[held]
        comp = apply_library(res.library, orig)
        assert expand(comp, res.library).key == orig.key, "leave-one-out round trip failed"
        s_orig = cost.size(orig)
        rows[held] = {
            "orig": s_orig,
            "loo_comp": cost.size(comp), "loo_saving": s_orig - cost.size(comp),
            "in_comp": cost.size(full.rewritten[held]), "in_saving": s_orig - cost.size(full.rewritten[held]),
            # the same savings in real cl100k tokens, whatever objective the library was learned with
            "orig_tok": ntok(orig.key),
            "loo_tok_saving": ntok(orig.key) - ntok(comp.key),
            "in_tok_saving": ntok(orig.key) - ntok(full.rewritten[held].key),
            "abstractions_learned": len(res.library),
            "abstractions_used": len({n.val for _, n in iter_sites(comp) if n.kind == "C"}),
        }
    return {"full": full, "rows": rows}


def transfer(corpus: dict, train_ids: list[str], objective: str = "nodes", **kw) -> dict:
    """Learn on ``train_ids`` only, apply to every other spec."""
    cost = Cost(objective)
    res = compress({k: corpus[k] for k in train_ids}, objective, **kw)
    rows = {}
    for k in sorted(corpus):
        orig = corpus[k] if isinstance(corpus[k], Node) else from_json(corpus[k])
        comp = apply_library(res.library, orig)
        assert expand(comp, res.library).key == orig.key
        rows[k] = {"train": k in train_ids, "orig": cost.size(orig), "comp": cost.size(comp),
                   "saving": cost.size(orig) - cost.size(comp),
                   "orig_tok": ntok(orig.key), "tok_saving": ntok(orig.key) - ntok(comp.key)}
    return {"result": res, "rows": rows}
