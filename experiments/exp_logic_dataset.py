"""exp_logic_dataset.py -- build reality-gated LOGIC datasets for the fine-tune study.

Two task families, both labeled by an EXACT gate (Z3 via newaxiom.contradiction), $0, no GPU, no API:

  valid  -- propositional ARGUMENT VALIDITY (does the conclusion follow?). This is proof-forming:
            an argument  P1..Pn |- C  is VALID iff  {P1,...,Pn, ~C}  is UNSATISFIABLE.
  consist -- joint CONSISTENCY (can all statements be true at once?) = SAT vs UNSAT.

The argument/clause SCHEMAS are only PROPOSERS; the Z3 gate DISPOSES -- every instance's label is the
gate's decision, never the schema's declared intent (reality-gating, honest tiering).

Held-out generalization: atoms are random distinct letters, so train and test share FORMS but never the
same (form, atom-assignment) instance -- the model must learn the logical FORM, not memorize a prompt.

writes: results/logic_dataset.json  {task: {train:[{prompt,label}], test:[...], meta:{...}}}
"""
from __future__ import annotations

import json
import os
import random
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from axuniv.newaxiom.contradiction import check_consistency, _Z3_OK  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "logic_dataset.json")
LETTERS = list(string.ascii_uppercase)          # A..Z -- abstract atoms, no world-knowledge leakage
N_TEST_PER_CLASS = 120
N_TRAIN_PER_CLASS = 140                          # pool; the worker subsamples 120/seed

# ---- propositional argument schemas:
#      (name, premise_formulas, concl_formula, premise_texts, concl_text, n_atoms)
# formulas use the contradiction parser syntax (~ & | ->); texts are the human rendering.
VALID_SCHEMAS = [
    ("modus_ponens",  ["{A} -> {B}", "{A}"],  "{B}",        ["if {A} then {B}", "{A}"],      "{B}",        2),
    ("modus_tollens", ["{A} -> {B}", "~{B}"], "~{A}",       ["if {A} then {B}", "not {B}"],  "not {A}",    2),
    ("hyp_syllogism", ["{A} -> {B}", "{B} -> {C}"], "{A} -> {C}",
                      ["if {A} then {B}", "if {B} then {C}"], "if {A} then {C}", 3),
    ("disj_syllogism",["{A} | {B}", "~{A}"],  "{B}",        ["{A} or {B}", "not {A}"],       "{B}",        2),
    ("constr_dilemma",["({A} -> {B}) & ({C} -> {D})", "{A} | {C}"], "{B} | {D}",
                      ["if {A} then {B}, and if {C} then {D}", "{A} or {C}"], "{B} or {D}", 4),
    ("contraposition",["{A} -> {B}"],         "~{B} -> ~{A}", ["if {A} then {B}"],  "if not {B} then not {A}", 2),
]
INVALID_SCHEMAS = [
    ("affirm_consequent",["{A} -> {B}", "{B}"], "{A}",      ["if {A} then {B}", "{B}"],      "{A}",        2),
    ("deny_antecedent",  ["{A} -> {B}", "~{A}"],"~{B}",     ["if {A} then {B}", "not {A}"],  "not {B}",    2),
    ("affirm_disjunct",  ["{A} | {B}", "{A}"],  "~{B}",     ["{A} or {B}", "{A}"],           "not {B}",    2),
    ("fake_chain",       ["{A} -> {B}", "{C} -> {D}"], "{A} -> {D}",
                      ["if {A} then {B}", "if {C} then {D}"], "if {A} then {D}", 4),
    ("converse_error",   ["{A} -> {B}"],        "{B} -> {A}", ["if {A} then {B}"],  "if {B} then {A}",    2),
    ("illicit_conj",     ["{A} | {B}"],         "{A} & {B}", ["{A} or {B}"],        "{A} and {B}",         2),
]


def _subst(tpl, mapping):
    for k, v in mapping.items():
        tpl = tpl.replace("{%s}" % k, v)
    return tpl


def _is_valid(prem_formulas, concl_formula, mapping):
    """EXACT gate: argument is valid iff premises + negated-conclusion is UNSAT."""
    clauses = [("p%d" % i, _subst(f, mapping)) for i, f in enumerate(prem_formulas)]
    clauses.append(("negC", "~(%s)" % _subst(concl_formula, mapping)))
    r = check_consistency(clauses)
    return r["status"] == "VERIFIED"        # VERIFIED == UNSAT == the negation is impossible == valid


def _render_argument(prem_texts, concl_text, mapping):
    prems = "; ".join("(%d) %s" % (i + 1, _subst(t, mapping)) for i, t in enumerate(prem_texts))
    concl = _subst(concl_text, mapping)
    return ("Premises: %s. Conclusion: %s.\nDoes the conclusion logically follow from the premises "
            "(is the argument valid)? Reply with only 'Yes' or 'No'." % (prems, concl))


def gen_validity(rng, need_per_class):
    """Return (pos, neg) lists of {prompt,label}; labels are the Z3 gate decision (not the schema tag)."""
    pos, neg, seen = [], [], set()
    schemas = [("valid", s) for s in VALID_SCHEMAS] + [("invalid", s) for s in INVALID_SCHEMAS]
    tries = 0
    while (len(pos) < need_per_class or len(neg) < need_per_class) and tries < 400_000:
        tries += 1
        _tag, (name, pf, cf, pt, ct, natoms) = rng.choice(schemas)
        atoms = rng.sample(LETTERS, natoms)
        mapping = {k: atoms[i] for i, k in enumerate(["A", "B", "C", "D"][:natoms])}
        prompt = _render_argument(pt, ct, mapping)
        if prompt in seen:
            continue
        seen.add(prompt)
        label = "Yes" if _is_valid(pf, cf, mapping) else "No"
        (pos if label == "Yes" else neg).append({"prompt": prompt, "label": label})
    return pos, neg


# ---- consistency: build inconsistent skeletons + random consistent sets; Z3 decides the label.
CLAUSE_TEXT = {"atom": "{A} is true", "natom": "{A} is false", "imp": "if {A} then {B}",
               "orr": "{A} or {B}", "andd": "{A} and {B}"}
CLAUSE_FORM = {"atom": "{A}", "natom": "~{A}", "imp": "{A} -> {B}", "orr": "{A} | {B}", "andd": "{A} & {B}"}
UNSAT_SKELETONS = [
    [("atom", "A"), ("natom", "A"), ("imp", "BC")],                 # A and not-A (+ distractor)
    [("imp", "AB"), ("atom", "A"), ("natom", "B")],                 # modus-ponens violation
    [("orr", "AB"), ("natom", "A"), ("natom", "B")],                # A|B but neither
    [("imp", "AB"), ("imp", "BC"), ("atom", "A"), ("natom", "C")],  # broken chain
]


def _build_clause(kind, atoms, tbl):
    m = {"A": atoms[0]}
    if len(atoms) > 1:
        m["B"] = atoms[1]
    if len(atoms) > 2:
        m["C"] = atoms[2]
    return _subst(tbl[kind], m)


def _consistency_prompt(clause_kinds_atoms):
    parts = []
    for i, (kind, atoms) in enumerate(clause_kinds_atoms):
        parts.append("(%d) %s" % (i + 1, _build_clause(kind, atoms, CLAUSE_TEXT)))
    return ("Statements: %s. Can all of these be true at the same time? Reply with only 'Yes' or 'No'."
            % "; ".join(parts))


def gen_consistency(rng, need_per_class):
    pos, neg, seen = [], [], set()   # pos = consistent (Yes), neg = inconsistent (No)
    kinds = ["atom", "natom", "imp", "orr"]
    tries = 0
    while (len(pos) < need_per_class or len(neg) < need_per_class) and tries < 400_000:
        tries += 1
        make_unsat = rng.random() < 0.5
        if make_unsat:
            skel = rng.choice(UNSAT_SKELETONS)
            names = rng.sample(LETTERS, 3)
            amap = {"A": names[0], "B": names[1], "C": names[2]}
            spec = [(k, [amap[c] for c in slot]) for k, slot in skel]
        else:
            n = rng.randint(3, 4)
            spec = []
            for _ in range(n):
                k = rng.choice(kinds)
                na = 2 if k in ("imp", "orr") else 1
                spec.append((k, rng.sample(LETTERS, na)))
        prompt = _consistency_prompt(spec)
        if prompt in seen:
            continue
        seen.add(prompt)
        clauses = [("c%d" % i, _build_clause(k, a, CLAUSE_FORM)) for i, (k, a) in enumerate(spec)]
        r = check_consistency(clauses)
        if r["status"] == "VERIFIED":
            label = "No"                       # UNSAT -> cannot all be true
        elif r["status"] == "ABSTAIN":
            label = "Yes"                      # SAT -> consistent
        else:
            continue                           # parse error -> discard, never guess
        (pos if label == "Yes" else neg).append({"prompt": prompt, "label": label})
    return pos, neg


def split(pos, neg):
    test = pos[:N_TEST_PER_CLASS] + neg[:N_TEST_PER_CLASS]
    random.Random(100).shuffle(test)
    train = pos[N_TEST_PER_CLASS:N_TEST_PER_CLASS + N_TRAIN_PER_CLASS] + \
        neg[N_TEST_PER_CLASS:N_TEST_PER_CLASS + N_TRAIN_PER_CLASS]
    return test, train


def main():
    if not _Z3_OK:
        print("ERROR: z3 unavailable -- cannot build gate-verified logic labels."); sys.exit(1)
    rng = random.Random(7)
    out = {}
    need = N_TEST_PER_CLASS + N_TRAIN_PER_CLASS + 20
    for task, gen in (("valid", gen_validity), ("consist", gen_consistency)):
        pos, neg = gen(rng, need)
        test, train = split(pos, neg)
        out[task] = {"train": train, "test": test,
                     "meta": {"n_pos": len(pos), "n_neg": len(neg),
                              "n_train": len(train), "n_test": len(test)}}
        print("%-8s pos=%d neg=%d -> train=%d test=%d" % (task, len(pos), len(neg), len(train), len(test)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.abspath(OUT))
    # show a couple of gate-labeled examples for the record
    for task in out:
        ex = out[task]["test"][0]
        print("\n[%s] %s\n   -> label(gate): %s" % (task, ex["prompt"].replace("\n", " "), ex["label"]))


if __name__ == "__main__":
    main()
