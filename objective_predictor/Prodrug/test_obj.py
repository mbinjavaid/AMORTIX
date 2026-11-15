from objective_predictor.Prodrug.bbb_obj import BBBObjective
from rdkit import Chem

bbb_objective = BBBObjective(qed_floor=0.4)

# (parent, prodrug) pairs
test_cases = [
    ("CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O", "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)OC"),  # ibuprofen methyl ester
    ("NCC(=O)O", "NCC(=O)OCC"),  # glycine ethyl ester
    ("CC(=O)NC1=CC=C(C=C1)O", "CC(=O)NC1=CC=C(C=C1)OC"),  # methyl ether, not cleavable
]

for parent, prodrug in test_cases:
    result = bbb_objective.calculate(Chem.MolFromSmiles(prodrug), Chem.MolFromSmiles(parent))
    m = result["metrics"]
    print(f"{prodrug}: reward={result['total_reward']:.3f} "
          f"bbb={m['bbb_prob']:.3f} qed={m['qed']:.2f} purity={m['purity']:.2f}")
