# Library learning (MDL) spike: results

Question: can the "few blocks" of the Deterministic Coder app spec language be LEARNED from working
specs by minimising `L(library) + L(corpus | library)`, instead of being written by hand, and does what
is learned generalise to a spec it never saw?

Status: a working, tested implementation (`symlang/library/`, 81 tests in `tests/test_library.py`).
**n = 6, so this is an existence check, not a rate.** Reproduce: `./.venv/bin/python -m symlang.library`
(and `--objective tokens`). Everything below is that command's output.

## What was built

Data: the six kernel presets (`counter`, `todo`, `bmi-calculator`, `tic-tac-toe`, `snake`,
`landing-page`) dumped from `deterministic-coder/ui/public/backend/kernel_presets.js` by
`symlang/library/dump_presets.js` (vm + fake global, as the repo's own test does) to
`symlang/library/data/presets.json`. 2445 nodes, 6674 cl100k tokens of minified JSON in total.

Algorithm (`symlang/library/learner.py`, stdlib + tiktoken only):

1. Trees are immutable; equality is on a canonical text, so `True != 1` and `1 != 1.0`, and dict key
   order is part of the tree (needed for an exact round trip). Size in nodes: leaf 1, dict 1 + #keys,
   list 1, call 1, children added. Size in tokens: cl100k tokens of the minified text (a call is
   written `@f3(a,b)`).
2. Candidates by anti-unification: the least general generalisation of every pair of same-shape
   subtrees (dict with the same key sequence, list of the same length, call of the same abstraction).
   Disagreements become numbered holes `?0 ?1 ...`, a repeated disagreement pair shares one hole.
   One refinement round widens the top patterns to cover further subtrees. Exactly repeated subtrees
   are zero-argument candidates.
3. Stitch-style greedy loop: rank candidates by estimated utility
   `sum over non-overlapping matches (size(match) - size(call)) - size(definition)`, then for the
   best ones actually rewrite the corpus and earlier definitions (top-down, arguments rewritten
   recursively) and **measure** utility as total-before minus total-after. Only a measured positive
   utility is accepted and reported. Stop when none is positive.
4. `expand(rewritten, library) == original` exactly, tested on the six presets and on 50 seeded random
   corpora (both objectives), plus held-out trees compressed with a learned library.

Prior art this follows (from memory, not re-verified in this spike): Stitch, Babble, DreamCoder
(library learning by compression), and Re-Pair / Sequitur (grammar compression). Check the papers
before citing.

## In-sample: library learned on all six (objective: nodes)

37 abstractions. Gross = corpus rewritten with calls, library not charged.

| spec | orig nodes | comp nodes | saved | orig tok | comp tok | saved |
|---|---:|---:|---:|---:|---:|---:|
| bmi-calculator | 418 | 223 | 46.7% | 1298 | 1054 | 18.8% |
| counter | 156 | 96 | 38.5% | 352 | 300 | 14.8% |
| landing-page | 548 | 332 | 39.4% | 1344 | 1109 | 17.5% |
| snake | 533 | 355 | 33.4% | 1631 | 1479 | 9.3% |
| tic-tac-toe | 339 | 231 | 31.9% | 1009 | 927 | 8.1% |
| todo | 451 | 253 | 43.9% | 1040 | 793 | 23.8% |
| **total, gross** | 2445 | 1490 | **39.1%** | 6674 | 5662 | **15.2%** |
| library | 370 nodes | | | 1167 tokens | | |
| **total, net (MDL)** | 1860 | | **23.9%** | 6829 | | **-2.3%** |

In nodes the MDL total shrinks by 24%. In real tokens, the same library is **not** a net win on six
specs: it saves 1012 tokens on the corpus and costs 1167 to write down. Node count flatters it,
because JSON structure (`{"set":..,"to":..}`, `{"class":..,"kids":..}`) is many nodes but few BPE
tokens (key names are 1-2 tokens each).

Learning with the token objective directly (14 abstractions, library 531 tokens): gross 10.8%
(6674 -> 5952), net **2.9%** (6483). The token-optimal library is smaller and more selective.

## Leave-one-out (learn on 5, compress the held-out 6th)

Library learned with the node objective. Savings gross (library is shared, not charged).

| held out | orig nodes | LOO saved | in-sample saved | LOO tokens saved | in-sample tokens saved | used / learned |
|---|---:|---:|---:|---:|---:|---:|
| bmi-calculator | 418 | 75 (17.9%) | 195 (46.7%) | 49 | 244 | 9/32 |
| counter | 156 | 46 (29.5%) | 60 (38.5%) | 33 | 52 | 8/35 |
| landing-page | 548 | 64 (11.7%) | 216 (39.4%) | 52 | 235 | 9/28 |
| snake | 533 | 122 (22.9%) | 178 (33.4%) | 73 | 152 | 14/33 |
| tic-tac-toe | 339 | 70 (20.6%) | 108 (31.9%) | 39 | 82 | 11/31 |
| todo | 451 | 80 (17.7%) | 198 (43.9%) | 61 | 247 | 13/31 |
| **total** | 2445 | **457 (18.7%)** | 955 (39.1%) | **307 (4.6% of 6674)** | 1012 (15.2%) | |

Plain reading: the library does NOT only memorise (held-out saving is far above 0), but it keeps
only about 48% of the in-sample saving in nodes and **30% in real tokens**. About half the
in-sample gain is overfit to the specs it was learned from.

With the token objective the library is nearly pure memorisation: held-out token saving 74 of 6674
(**1.1%**) vs 722 in-sample (10.8%); each held-out spec uses a single abstraction.

Train on the three form/list presets (counter, todo, bmi), apply to the other three (node objective,
18 abstractions): landing-page 9.1% (49 tokens), snake 18.8% (67 tokens), tic-tac-toe 15.0%
(25 tokens). With the token objective this transfer is exactly 0 (only 2 abstractions are learned
from three specs, both specific to the form field and chip button).

## Top learned abstractions (node objective, flat meaning, `?i` = argument)

Judge by eye whether they are blocks. Four are, the rest are record shapes:

- `f0(?0,?1,?2)` **form field**: `{class:"field", kids:[{tag:"label", attrs:{for:?0}, text:?1},
  {tag:"input", id:?0, bind:?2, attrs:{type:"text", inputmode:"decimal", autocomplete:"off"}}]}`.
  4 uses (bmi). Label tied to its input by id, bound to a state name. A real block, and the
  single most valuable abstraction in both objectives (93 tokens).
- `f4(?0..?3)` **filter chip button**: `{tag:"button", id, class:"chip", attrs:{type:"button",
  aria-pressed:?1}, on:{click:[{set:"show", to:?2}]}, text}`. 3 uses (todo). The learner re-found the
  hand-written `filterBtn` helper in `kernel_presets.js` without being told it exists.
- `f7(?0..?6)` **inline form**: `<form class="row" autocomplete=off>` with sr-only label, input bound
  to state, primary submit button. 2 uses (todo, landing-page).
- `f3(?0,?1,?2)` **guarded state-set rule** `{set, to, when}` (19 uses) and `f2(?0,?1)` **state-set**
  `{set, to}` (28 uses): the "increment/assign a state slot" action, the most frequent unit.
- `f1(?0..?4)` **element** `{tag, id, class, attrs, text}` (13 uses), and `f6(?0,?1)` **event
  handler** `{on, do:[?1]}` (11 uses).

What it did NOT find: a "list with delete button" block (the todo row), or "button bound to a state
increment" as a single unit (counter has `inc`/`dec`, but the click rules and the buttons live in
different parts of the spec, so no single subtree contains both). Both are cross-subtree patterns
that a subtree-based learner cannot express. Several abstractions (`{class,kids}`, `{tag,id,text}`)
are bare record shapes that compress key names, not blocks; they are what makes the node-objective
numbers look better than the token numbers.

## Limits, said plainly

- **n = 6.** Presets are hand-written by one author in one style, and share infrastructure. 6 folds of
  leave-one-out is an existence check that generalisation is above zero, not a rate. Do not read
  18.7% or 4.6% as expected values on new specs.
- Strings are atomic: `"faq-q{i + 1}"`, `"click:#inc"`, `"clamp(count + 1, -999999, 999999)"` are not
  decomposed, so expression-language idioms (the biggest token cost) are not abstracted at all.
- Dicts with different key sets do not unify (an element with and without `class` is a hole at the
  root, not a shared shape with an optional key).
- Only subtree patterns, no cross-subtree or name-binding abstraction (the counter "button + rule" case).
- The library must be shown to the model: 1167 tokens (node library) or 531 (token library). At the
  held-out rate (51 and 12 tokens saved per spec) the break-even is roughly 23 and 43 specs of this
  size in one prompt context, unless the macros are expanded by a deterministic system the model
  never has to read (names carried in a fine-tune, or looked up on demand). Not tested.
- Tokens were counted on a text form (`@f3(a,b)`) that is ours. A model writing it has to learn the
  syntax and may make errors the JSON form would not: no model was run in this spike.
- Greedy, not optimal; the accepted order matters. Ranking uses an estimate, but every claimed saving
  is re-measured (the `telescopes` test checks that the step utilities sum to the real before/after).

## What corpus growth would be needed

The Deterministic Coder repo also has 24 recorded model replies, the 24 hand-written archetype
templates and the held-out specs. Re-expressing those as kernel specs gives a corpus of around 30-50
specs, which is the first size where leave-one-out becomes a rate with a visible variance (do
10-fold, report the spread, learn on tokens, charge the library). Conditions to scale: (1) add
sub-expression anti-unification inside strings (parse the expression language already in
`kernel_expr.js`), (2) allow optional dict keys, (3) learn on model-written specs too, so the
library reflects what the model emits and not what the author wrote, (4) hold out whole spec
families, not single specs, since presets of one family share structure.

## Verdict

MDL library learning works as a mechanism (exact round trip, measured savings, rediscovers a
hand-written helper and a real form-field block, 37 abstractions from 6 specs in 0.2 s) but the
evidence for the token claim is weak: net -2.3% (node library) or +2.9% (token library) in-sample,
and 1.1-4.6% held-out gross. It is worth scaling the data, not the claim: grow the corpus to
30+ specs and learn with the token objective before expecting the model to write fewer tokens.
