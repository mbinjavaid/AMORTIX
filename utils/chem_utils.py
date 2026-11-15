from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestRegressor


class SurrogateModel:
    """
    A lightweight Random Forest Regressor acting as a cheap stand-in for the oracle.
    It predicts the continuous objective score of a molecule from its Morgan fingerprint and is used to decide which of
    the sampled molecules are worth spending real oracle calls on.
    """

    def __init__(self, n_jobs: int = -1):
        self.model = RandomForestRegressor(
            n_estimators=200,
            n_jobs=n_jobs,
            max_depth=None,
            min_samples_leaf=1,
            max_features=1 / 3
        )
        self.is_fitted = False
        self.fp_cache = {}

    def _get_fp(self, smiles: str) -> Optional[np.ndarray]:
        if smiles in self.fp_cache:
            return self.fp_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        self.fp_cache[smiles] = arr
        return arr

    def train(self, smiles_list: List[str], scores_list: List[float]):
        """Trains the regressor on the raw scores of all (unique, valid) molecules seen so far."""
        if not smiles_list:
            return

        X = []
        valid_scores = []
        seen_smiles = set()

        for smi, score in zip(smiles_list, scores_list):
            if score == float("-inf") or score is None or smi is None:
                continue
            if smi in seen_smiles:
                continue
            fp = self._get_fp(smi)
            if fp is not None:
                X.append(fp)
                valid_scores.append(score)
                seen_smiles.add(smi)

        if len(X) < 50:
            print(f"[Surrogate] Not enough data to train yet ({len(X)} samples).")
            return

        y = np.array(valid_scores)

        try:
            self.model.fit(X, y)
            self.is_fitted = True
            r2 = self.model.score(X, y)
            print(f"[Surrogate] Trained on {len(X)} unique samples. R2 Score: {r2:.3f}")
        except Exception as e:
            print(f"[Surrogate] Training failed: {e}")
            self.is_fitted = False

        self.fp_cache = {}

    def predict_ranking_scores(self, smiles_list: List[str]) -> np.ndarray:
        """Returns the predicted score of each molecule (0 for molecules that cannot be parsed)."""
        if not self.is_fitted:
            return np.random.rand(len(smiles_list))

        X = []
        indices_kept = []
        for i, smi in enumerate(smiles_list):
            if smi is None:
                continue
            fp = self._get_fp(smi)
            if fp is not None:
                X.append(fp)
                indices_kept.append(i)

        if not X:
            return np.zeros(len(smiles_list))

        preds = self.model.predict(X)

        final_scores = np.zeros(len(smiles_list))
        for idx, p in zip(indices_kept, preds):
            final_scores[idx] = p
        return final_scores
