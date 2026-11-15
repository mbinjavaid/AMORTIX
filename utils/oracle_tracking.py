"""
Oracle bookkeeping: counts unique oracle calls (by canonical SMILES), enforces a budget, writes every call to
`oracle_history.csv` and tracks the top-10 mean (`top10_curve.csv`) and its AUC over oracle calls.

`OracleLedger` is the pure-python core, `OracleTracker` its Ray actor (used by `main.py`). `CentralOracle` additionally
scores with a TDC oracle (used by the PMO runner).
"""
import csv
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem, RDLogger


def top10_auc(curve_points: Sequence[Tuple[int, float]], n_calls: int, max_budget: int = 10000,
              freq: int = 100) -> float:
    """Top-10 AUC from curve points [(unique_calls, top10_mean), ...]: sampled every `freq` calls (trapezoidal rule),
    extended flat to `max_budget` and divided by `max_budget`."""
    n_calls = min(int(n_calls), max_budget)
    if not curve_points:
        return 0.0

    grid_x = list(range(0, n_calls + 1, freq))
    if grid_x[-1] != n_calls:
        grid_x.append(n_calls)

    raw_x = [p[0] for p in curve_points]
    raw_y = [p[1] for p in curve_points]
    indices = np.searchsorted(raw_x, grid_x, side="right") - 1
    grid_y = [raw_y[i] if i >= 0 else 0.0 for i in indices]

    if grid_x[-1] < max_budget:
        grid_x.append(max_budget)
        grid_y.append(grid_y[-1])

    area = 0.0
    for i in range(1, len(grid_x)):
        area += (grid_x[i] - grid_x[i - 1]) * (grid_y[i] + grid_y[i - 1]) / 2.0
    return float(area / max_budget)


Item = Union[str, Tuple[str, Optional[str]]]  # SMILES, or (SMILES, context SMILES)


def _split(item: Item) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(item, (tuple, list)):
        return item[0], (item[1] if len(item) > 1 else None)
    return item, None


def canonical_key(item: Item) -> Optional[Tuple[str, Optional[str]]]:
    """(canonical SMILES, canonical context or None); None for empty / invalid SMILES and for the trivial "C"."""
    smiles, context = _split(item)
    if not smiles:
        return None
    try:
        canon = Chem.CanonSmiles(smiles)
    except Exception:
        return None
    if canon == "C":
        return None
    if context:
        try:
            context = Chem.CanonSmiles(context)
        except Exception:
            pass
    else:
        context = None
    return canon, context


class OracleLedger:
    HISTORY_FILE = "oracle_history.csv"
    TOP10_FILE = "top10_curve.csv"

    def __init__(self, results_dir: Optional[str] = None, max_budget: Optional[int] = 10000, freq: int = 100):
        RDLogger.DisableLog('rdApp.*')
        self.max_budget = float("inf") if max_budget is None else max_budget
        self.freq = freq

        self.global_cache: Dict[Tuple[str, Optional[str]], float] = {}  # key -> score
        self.aux_cache: Dict[Tuple[str, Optional[str]], dict] = {}  # key -> extra metrics of the objective, if any
        self.unique_calls = 0
        self.budget_exceeded = False

        # (unique_calls, top-10 mean) step-function points
        self.auc_curve_points: List[Tuple[int, float]] = [(0, 0.0)]
        self.all_valid_molecules: List[Tuple[float, str]] = []  # (score, smiles), trimmed to the best 200

        self.history_path = self.top10_path = None
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            self.history_path = os.path.join(results_dir, self.HISTORY_FILE)
            self.top10_path = os.path.join(results_dir, self.TOP10_FILE)
            with open(self.history_path, "w", newline="") as f:
                csv.writer(f).writerow(["call_id", "smiles", "score", "context"])
            with open(self.top10_path, "w", newline="") as f:
                csv.writer(f).writerow(["unique_calls", "top10_avg", "top10_smiles_scores"])

    # ------------------------------------------------------------------ top-10 tracking
    def _update_top10_tracker(self):
        self.all_valid_molecules.sort(key=lambda x: x[0], reverse=True)
        if len(self.all_valid_molecules) > 200:
            self.all_valid_molecules = self.all_valid_molecules[:200]

        top_10 = self.all_valid_molecules[:10]
        if not top_10:
            avg_score, mol_str = 0.0, ""
        else:
            avg_score = sum(x[0] for x in top_10) / len(top_10)
            mol_str = ";".join(f"{x[1]}:{x[0]:.4f}" for x in top_10)

        if self.unique_calls > self.auc_curve_points[-1][0]:
            self.auc_curve_points.append((self.unique_calls, avg_score))

        if self.top10_path is not None:
            with open(self.top10_path, "a", newline="") as f:
                csv.writer(f).writerow([self.unique_calls, f"{avg_score:.6f}", mol_str])

    def get_top10_avg(self) -> float:
        return self.auc_curve_points[-1][1]

    def get_best_score(self) -> float:
        if not self.all_valid_molecules:
            return float("-inf")
        return max(x[0] for x in self.all_valid_molecules)

    def get_top10_auc(self) -> float:
        """Top-10 AUC so far (flat extension to the budget; without a budget: normalised by the calls so far)."""
        budget = self.max_budget if self.max_budget != float("inf") else max(self.unique_calls, 1)
        return top10_auc(self.auc_curve_points, self.unique_calls, budget, self.freq)

    def get_budget_status(self):
        return self.unique_calls, self.budget_exceeded

    def get_count(self) -> int:
        return self.unique_calls

    def get_stats(self) -> dict:
        return {
            "unique_calls": self.unique_calls,
            "budget_exceeded": self.budget_exceeded,
            "top10_avg": self.get_top10_avg(),
            "best_score": self.get_best_score(),
            "top10_auc": self.get_top10_auc(),
        }

    # ------------------------------------------------------------------ lookup / record
    def lookup(self, items: List[Item]) -> list:
        """
        Says what has to happen with each molecule of a batch:
          ("skip",)            invalid / empty / trivial: result is -inf, not counted
          ("cached", score, aux)  already scored before
          ("new", key)         has to be scored now (first occurrence in this batch, budget left)
          ("dup", j)           same molecule as item j of this batch (j < own index)
          ("over",)            budget used up: result is -inf
        """
        remaining = self.max_budget - self.unique_calls
        first_in_batch: Dict[tuple, int] = {}
        out = []
        for i, item in enumerate(items):
            key = canonical_key(item)
            if key is None:
                out.append(("skip",))
            elif key in self.global_cache:
                out.append(("cached", self.global_cache[key], self.aux_cache.get(key)))
            elif key in first_in_batch:
                out.append(("dup", first_in_batch[key]))
            elif len(first_in_batch) < remaining:
                first_in_batch[key] = i
                out.append(("new", key))
            else:
                self.budget_exceeded = True
                out.append(("over",))
        return out

    def record(self, items: List[Item], scores: Sequence[float], extras: Optional[Sequence[Optional[dict]]] = None
               ) -> List[float]:
        """
        Stores the scores of molecules that were scored after `lookup` said "new". Returns the score of each item as
        counted by the ledger: the stored score if the molecule was recorded in the meantime (several workers),
        -inf if the budget is used up.
        """
        results = []

        self.all_valid_molecules.sort(key=lambda x: x[0], reverse=True)
        if len(self.all_valid_molecules) < 10:
            top10_threshold = float("-inf")
        else:
            top10_threshold = self.all_valid_molecules[9][0]

        history_rows = []
        for n, (item, score) in enumerate(zip(items, scores)):
            key = canonical_key(item)
            if key is None:
                results.append(float("-inf"))
                continue
            if key in self.global_cache:
                results.append(self.global_cache[key])
                continue
            if self.unique_calls >= self.max_budget:
                self.budget_exceeded = True
                results.append(float("-inf"))
                continue

            score = float(score)
            self.global_cache[key] = score
            if extras is not None and extras[n] is not None:
                self.aux_cache[key] = extras[n]
            self.unique_calls += 1
            canon_smi, context = key
            history_rows.append([self.unique_calls, canon_smi, score, context or ""])
            results.append(score)

            if score > float("-inf"):
                self.all_valid_molecules.append((score, canon_smi))
                # A molecule that beats the 10th best changes the top-10 mean: store a new curve point right away.
                if score > top10_threshold:
                    self._update_top10_tracker()
                    if len(self.all_valid_molecules) < 10:
                        top10_threshold = float("-inf")
                    else:
                        top10_threshold = self.all_valid_molecules[9][0]

        if history_rows and self.history_path is not None:
            with open(self.history_path, "a", newline="") as f:
                csv.writer(f).writerows(history_rows)

        return results


class PMOOracleCore(OracleLedger):
    """Ledger plus scoring with `score_fn(canonical_smiles) -> float` (e.g. a TDC oracle)."""

    def __init__(self, score_fn: Callable[[str], float], results_dir: Optional[str] = None,
                 max_budget: Optional[int] = 10000, freq: int = 100):
        super().__init__(results_dir, max_budget, freq)
        self.score_fn = score_fn

    def predict_batch(self, smiles_list: List[Item]) -> List[float]:
        statuses = self.lookup(smiles_list)
        new_pos = [i for i, s in enumerate(statuses) if s[0] == "new"]
        new_scores = [float(self.score_fn(statuses[i][1][0])) for i in new_pos]
        recorded = self.record([smiles_list[i] for i in new_pos], new_scores)

        results = [float("-inf")] * len(smiles_list)
        for k, i in enumerate(new_pos):
            results[i] = recorded[k]
        for i, s in enumerate(statuses):
            if s[0] == "cached":
                results[i] = s[1]
            elif s[0] == "dup":
                results[i] = results[s[1]]
        return results


try:
    import ray
except ImportError:  # the pure-python classes above stay usable without ray
    ray = None

if ray is not None:

    @ray.remote
    class OracleTracker(OracleLedger):
        """Ray actor: one shared ledger (cache / budget / history / top-10) for all rollout workers."""

    @ray.remote
    class CentralOracle:
        """Ray actor: one shared ledger around a TDC oracle for all rollout workers (PMO runner)."""

        def __init__(self, task_name: str, results_dir: str, max_budget: int = 10000, freq: int = 100):
            from tdc import Oracle  # imported lazily: only needed on the machine that runs the oracle
            self.core = PMOOracleCore(Oracle(name=task_name), results_dir, max_budget, freq)

        def predict_batch(self, smiles_list: List[str]) -> List[float]:
            return self.core.predict_batch(smiles_list)

        def get_budget_status(self):
            return self.core.get_budget_status()

        def get_top10_auc(self) -> float:
            return self.core.get_top10_auc()

        def get_stats(self) -> dict:
            return self.core.get_stats()


class RemoteEvaluatorProxy:
    """Drop-in replacement for `MoleculeObjectiveEvaluator` inside rollout workers: forwards to the central oracle."""

    def __init__(self, actor_handle):
        self.actor = actor_handle

    def predict_objective(self, molecule_designs: List[Union["MoleculeDesign", str]]) -> np.ndarray:
        from molecule_design import MoleculeDesign

        smiles_input = []
        for mol in molecule_designs:
            if isinstance(mol, MoleculeDesign):
                if mol.synthesis_done and not mol.infeasibility_flag:
                    smiles_input.append(mol.smiles_string)
                else:
                    smiles_input.append("")
            else:
                smiles_input.append(mol)

        scores = ray.get(self.actor.predict_batch.remote(smiles_input))

        for i, mol in enumerate(molecule_designs):
            if isinstance(mol, MoleculeDesign):
                mol.objective = scores[i]

        return np.array(scores)
