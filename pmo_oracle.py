"""
Oracle bookkeeping for PMO-style (practical molecular optimization) benchmarking.

Rules implemented here:
  * Molecules are identified by their canonical SMILES; only *unique* molecules consume oracle budget.
  * Invalid SMILES, empty SMILES and the trivial molecule "C" return -inf and are not counted.
  * Once `max_budget` unique molecules have been scored, every new (uncached) molecule returns -inf.
  * Whenever a new molecule enters the running top-10, the pair (unique_calls, mean of current top-10) is stored.
    These pairs form a step function over the number of oracle calls.
  * The top-10 AUC is the area under that step function, sampled every `freq` calls (trapezoidal rule), extended
    flat up to `max_budget` and divided by `max_budget`. This matches `top_auc(buffer, 10, finish=True, freq_log=100,
    max_oracle_calls=10000)` of the PMO benchmark (mol_opt).

`PMOOracleCore` holds all logic and has no Ray / TDC dependency. `CentralOracle` wraps it as a Ray actor around a TDC
oracle so that all rollout workers share one cache, one budget and one top-10 tracker.
"""
import csv
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem, RDLogger


def top10_auc(curve_points: Sequence[Tuple[int, float]], n_calls: int, max_budget: int = 10000,
              freq: int = 100) -> float:
    """
    Top-10 AUC from the step-function curve points [(unique_calls, top10_mean), ...] (first point must be (0, 0.0)).

    The curve is evaluated on the grid 0, freq, 2*freq, ..., <= n_calls plus the exact point n_calls (holding the
    previous value between points), then extended flat to `max_budget`, integrated with the trapezoidal rule and
    divided by `max_budget`. Can be called at any time during a run.
    """
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


class PMOOracleCore:
    HISTORY_FILE = "oracle_history.csv"
    TOP10_FILE = "pmo_top10_curve.csv"

    def __init__(self, score_fn: Callable[[str], float], results_dir: Optional[str] = None, max_budget: int = 10000,
                 freq: int = 100):
        RDLogger.DisableLog('rdApp.*')
        self.score_fn = score_fn
        self.max_budget = max_budget
        self.freq = freq

        self.global_cache: Dict[str, float] = {}  # canonical SMILES -> score
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
                csv.writer(f).writerow(["call_id", "smiles", "score"])
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

    def get_current_pmo_score(self) -> float:
        return top10_auc(self.auc_curve_points, self.unique_calls, self.max_budget, self.freq)

    def get_budget_status(self):
        return self.unique_calls, self.budget_exceeded

    # ------------------------------------------------------------------ scoring
    def predict_batch(self, smiles_list: List[str]) -> List[float]:
        results = []

        self.all_valid_molecules.sort(key=lambda x: x[0], reverse=True)
        if len(self.all_valid_molecules) < 10:
            top10_threshold = float("-inf")
        else:
            top10_threshold = self.all_valid_molecules[9][0]

        history_rows = []
        for smi in smiles_list:
            if not smi:
                results.append(float("-inf"))
                continue
            try:
                canon_smi = Chem.CanonSmiles(smi)
            except Exception:
                results.append(float("-inf"))
                continue
            if canon_smi == "C":
                results.append(float("-inf"))
                continue

            if canon_smi in self.global_cache:
                results.append(self.global_cache[canon_smi])
                continue

            if self.unique_calls >= self.max_budget:
                self.budget_exceeded = True
                results.append(float("-inf"))
                continue

            score = float(self.score_fn(canon_smi))
            self.global_cache[canon_smi] = score
            self.unique_calls += 1
            history_rows.append([self.unique_calls, canon_smi, score])
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


try:
    import ray
except ImportError:  # the pure-python core above stays usable without ray
    ray = None

if ray is not None:

    @ray.remote
    class CentralOracle:
        """Ray actor: one shared cache / budget / top-10 tracker around a TDC oracle for all rollout workers."""

        def __init__(self, task_name: str, results_dir: str, max_budget: int = 10000, freq: int = 100):
            from tdc import Oracle  # imported lazily: only needed on the machine that runs the oracle
            self.core = PMOOracleCore(Oracle(name=task_name), results_dir, max_budget, freq)

        def predict_batch(self, smiles_list: List[str]) -> List[float]:
            return self.core.predict_batch(smiles_list)

        def get_budget_status(self):
            return self.core.get_budget_status()

        def get_current_pmo_score(self) -> float:
            return self.core.get_current_pmo_score()

        def get_stats(self) -> dict:
            return {
                "unique_calls": self.core.unique_calls,
                "budget_exceeded": self.core.budget_exceeded,
                "top10_avg": self.core.get_top10_avg(),
                "best_score": self.core.get_best_score(),
                "pmo_auc": self.core.get_current_pmo_score(),
            }


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
