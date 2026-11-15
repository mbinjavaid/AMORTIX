"""
Checks the oracle bookkeeping and the top-10 AUC against a direct re-implementation of the PMO benchmark's
`top_auc(buffer, top_n=10, finish=True, freq_log=100, max_oracle_calls=10000)`.

Run with:  python -m pytest tests/test_pmo_oracle.py   (or: python tests/test_pmo_oracle.py)
"""
import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pmo_oracle import PMOOracleCore, top10_auc  # noqa: E402


def reference_top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    """buffer: {smiles: [score, call_index]} (PMO benchmark implementation)."""
    total = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        total += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    total += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        total += (max_oracle_calls - len(buffer)) * top_n_now
    return total / max_oracle_calls


def _molecules(n, rng):
    """n distinct valid SMILES (linear alkanes/alcohols with a varying pattern)."""
    mols = []
    i = 0
    while len(mols) < n:
        i += 1
        s = "C" * (1 + i % 40) + "O" + "N" * (i // 40 % 5) + "C" * (i // 200)
        if s not in mols:
            mols.append(s)
    rng.shuffle(mols)
    return mols


def _make_scorer(rng, with_ties):
    cache = {}

    def score(smi):
        if smi not in cache:
            cache[smi] = round(rng.random(), 1) if with_ties else rng.random()
        return cache[smi]
    return score


def _run_case(n_unique, with_ties, seed, max_budget=10000):
    rng = random.Random(seed)
    score = _make_scorer(rng, with_ties)
    with tempfile.TemporaryDirectory() as d:
        core = PMOOracleCore(score, d, max_budget=max_budget)
        # build SMILES that RDKit keeps distinct: use a mix of chain lengths and branching
        mols = []
        k = 0
        while len(mols) < n_unique:
            k += 1
            smi = "C" * (k % 30 + 2) + "(" + "C" * (k // 30 % 30 + 1) + ")" + "O" * 1 + "C" * (k // 900)
            mols.append(smi)
        # duplicates and invalid entries must not change anything
        seq = []
        for m in mols:
            seq.append(m)
            if rng.random() < 0.1:
                seq.append(rng.choice(mols[: len(seq)]))
            if rng.random() < 0.02:
                seq.append("not_a_smiles")
            if rng.random() < 0.02:
                seq.append("C")
            if rng.random() < 0.02:
                seq.append("")
        for i in range(0, len(seq), 7):  # arbitrary batching
            core.predict_batch(seq[i:i + 7])
        auc = core.get_current_pmo_score()
        buffer = {smi: [sc, i + 1] for i, (smi, sc) in enumerate(core.global_cache.items())}
        return core, auc, buffer


def test_matches_reference_auc():
    for n_unique, ties, seed in [(1, False, 0), (5, False, 1), (9, True, 2), (10, True, 3), (99, False, 4),
                                 (100, False, 5), (101, True, 6), (250, True, 7), (730, False, 8), (1500, True, 9)]:
        core, auc, buffer = _run_case(n_unique, ties, seed, max_budget=10000)
        assert core.unique_calls == len(buffer)
        ref = reference_top_auc(buffer, 10, True, 100, 10000)
        assert abs(auc - ref) < 1e-9, (n_unique, ties, auc, ref)


def test_budget_exactly_reached_and_overflow():
    core, auc, buffer = _run_case(60, False, 11, max_budget=50)
    assert core.unique_calls == 50 and core.budget_exceeded
    assert len(buffer) == 50
    ref = reference_top_auc(buffer, 10, True, 100, 50)
    assert abs(auc - ref) < 1e-9
    # molecules beyond the budget must return -inf, cached ones keep their score
    first = next(iter(core.global_cache))
    assert core.predict_batch([first]) == [core.global_cache[first]]
    assert core.predict_batch(["CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCO"]) == [float("-inf")]


def test_full_budget_of_10000():
    rng = random.Random(123)
    score = _make_scorer(rng, True)
    cands = ["".join(rng.choice("CNO") for _ in range(14)) for _ in range(14000)]
    with tempfile.TemporaryDirectory() as d:
        core = PMOOracleCore(score, d, max_budget=10000)
        for i in range(0, len(cands), 50):
            core.predict_batch(cands[i:i + 50])
        assert core.unique_calls == 10000 and core.budget_exceeded
        buffer = {smi: [sc, i + 1] for i, (smi, sc) in enumerate(core.global_cache.items())}
        ref = reference_top_auc(buffer, 10, True, 100, 10000)
        assert abs(core.get_current_pmo_score() - ref) < 1e-9


def test_trivial_and_invalid_not_counted():
    with tempfile.TemporaryDirectory() as d:
        core = PMOOracleCore(lambda s: 0.5, d)
        out = core.predict_batch(["C", "", "xyz", "CC", "OCC", "CCO"])  # "OCC" and "CCO" are the same molecule
        assert out[:3] == [float("-inf")] * 3
        assert out[3] == 0.5 and out[4] == 0.5 and out[5] == 0.5
        assert core.unique_calls == 2


def test_csv_history_reproduces_auc():
    """The AUC recomputed from oracle_history.csv alone must equal the tracker's AUC."""
    import csv
    rng = random.Random(5)
    score = _make_scorer(rng, False)
    with tempfile.TemporaryDirectory() as d:
        core = PMOOracleCore(score, d)
        mols = ["C" * (k % 30 + 2) + "(" + "C" * (k // 30 + 1) + ")O" for k in range(400)]
        for i in range(0, len(mols), 13):
            core.predict_batch(mols[i:i + 13])
        with open(os.path.join(d, "oracle_history.csv")) as f:
            rows = list(csv.DictReader(f))
        buffer = {r["smiles"]: [float(r["score"]), int(r["call_id"])] for r in rows}
        assert abs(reference_top_auc(buffer, 10, True, 100, 10000) - core.get_current_pmo_score()) < 1e-9


def test_top10_auc_function_flat_tail():
    # one molecule with score 1 at call 1, nothing else: area = 1 * (10000 - 1) / 10000 + triangle of first call
    pts = [(0, 0.0), (1, 1.0)]
    assert abs(top10_auc(pts, 1) - ((1 * 0.5) + 9999 * 1.0) / 10000) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
