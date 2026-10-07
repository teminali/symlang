"""Immutable tree representation for library learning.

A tree is built from JSON values (dict / list / str / int / float / bool / None) plus two
extra node kinds that only exist inside learned libraries:

    Hole(i)          a numbered parameter ``?i`` in an abstraction body
    Call(name, ...)  a use of a learned abstraction, ``@name(arg0,arg1,...)``

Every node carries a canonical text ``key``: for plain JSON it is exactly
``json.dumps(x, separators=(',', ':'), ensure_ascii=False)`` (so it is also the minified text the
token measure is taken on). Two trees are equal iff their keys are equal, so ``True`` and ``1``,
or ``1`` and ``1.0``, are different leaves and key order of dicts is part of the tree. This is
what makes ``expand(compress(x)) == x`` exact rather than "equal up to Python's loose ``==``".
"""
from __future__ import annotations

import json

_dump = lambda v: json.dumps(v, ensure_ascii=False)  # noqa: E731


class Node:
    __slots__ = ("kind", "val", "keys", "kids", "key", "nodes", "nh", "shape")

    def __init__(self, kind, val=None, keys=(), kids=()):
        self.kind = kind          # 'L' leaf, 'D' dict, 'S' list, 'H' hole, 'C' call
        self.val = val            # leaf value / hole index / call name
        self.keys = keys          # dict keys (ordered)
        self.kids = kids          # children (dict values / list items / call args)
        if kind == "L":
            self.key = _dump(val)
            self.nodes = 1
            self.nh = 0
            self.shape = None
        elif kind == "H":
            self.key = f"?{val}"
            self.nodes = 1
            self.nh = 1
            self.shape = None
        elif kind == "D":
            self.key = "{" + ",".join(f"{_dump(k)}:{c.key}" for k, c in zip(keys, kids)) + "}"
            self.nodes = 1 + len(keys) + sum(c.nodes for c in kids)
            self.nh = sum(c.nh for c in kids)
            self.shape = ("D", keys)
        elif kind == "S":
            self.key = "[" + ",".join(c.key for c in kids) + "]"
            self.nodes = 1 + sum(c.nodes for c in kids)
            self.nh = sum(c.nh for c in kids)
            self.shape = ("S", len(kids))
        elif kind == "C":
            self.key = f"@{val}(" + ",".join(c.key for c in kids) + ")"
            self.nodes = 1 + sum(c.nodes for c in kids)
            self.nh = sum(c.nh for c in kids)
            self.shape = ("C", val, len(kids))
        else:  # pragma: no cover
            raise ValueError(kind)

    def __eq__(self, other):
        return isinstance(other, Node) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return f"Node({self.key})"


def leaf(v) -> Node:
    return Node("L", v)


def hole(i: int) -> Node:
    return Node("H", i)


def mk_dict(keys, kids) -> Node:
    return Node("D", None, tuple(keys), tuple(kids))


def mk_list(kids) -> Node:
    return Node("S", None, (), tuple(kids))


def call(name: str, args) -> Node:
    return Node("C", name, (), tuple(args))


def from_json(x) -> Node:
    if isinstance(x, dict):
        for k in x:
            if not isinstance(k, str):
                raise TypeError(f"non-string dict key {k!r}")
        return mk_dict(list(x.keys()), [from_json(v) for v in x.values()])
    if isinstance(x, (list, tuple)):
        return mk_list([from_json(v) for v in x])
    if x is None or isinstance(x, (str, bool, int, float)):
        return leaf(x)
    raise TypeError(f"unsupported value {type(x).__name__}")


def to_json(n: Node):
    if n.kind == "L":
        return n.val
    if n.kind == "D":
        return {k: to_json(c) for k, c in zip(n.keys, n.kids)}
    if n.kind == "S":
        return [to_json(c) for c in n.kids]
    raise ValueError(f"tree still contains a {'hole' if n.kind == 'H' else 'call'}: {n.key[:60]}")


def with_kids(n: Node, kids) -> Node:
    """Same node with new children (identity preserved when nothing changed)."""
    if all(a is b for a, b in zip(kids, n.kids)):
        return n
    if n.kind == "D":
        return mk_dict(n.keys, kids)
    if n.kind == "S":
        return mk_list(kids)
    if n.kind == "C":
        return call(n.val, kids)
    raise ValueError(n.kind)


def iter_sites(root: Node):
    """Preorder (path, node) for every dict / list / call node. Path = child indices."""
    stack = [((), root)]
    while stack:
        path, n = stack.pop()
        if n.kind in "DSC":
            yield path, n
            for i in range(len(n.kids) - 1, -1, -1):
                stack.append((path + (i,), n.kids[i]))
