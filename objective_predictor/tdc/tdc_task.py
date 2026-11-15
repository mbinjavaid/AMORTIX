from tdc import Oracle

# Supported TDC oracle tasks ("jnk3" and "kinase_mpo" have their own classes).
TDC_TASKS = [
    "albuterol_similarity", "amlodipine_mpo", "aripiprazole_similarity", "celecoxib_rediscovery", "deco_hop",
    "drd2", "fexofenadine_mpo", "gsk3b", "isomers_c11h24", "isomers_c7h8n2o2", "isomers_c9h10n2o2pf2cl",
    "median1", "median2", "mestranol_similarity", "osimertinib_mpo", "perindopril_mpo", "qed", "ranolazine_mpo",
    "scaffold_hop", "sitagliptin_mpo", "thiothixene_rediscovery", "troglitazone_rediscovery", "valsartan_smarts",
    "zaleplon_mpo",
]

# Alternative task names
TASK_ALIASES = {
    "median_camphor_menthol": "median1",
    "median_tadalafil_sildenafil": "median2",
    "perindopril_rings": "perindopril_mpo",
    "amlodipine_rings": "amlodipine_mpo",
    "sitagliptin_replacement": "sitagliptin_mpo",
}


def resolve_task_name(name: str) -> str:
    return TASK_ALIASES.get(name, name)


class TDCTaskObjective:
    """Objective given by a TDC oracle (e.g., 'celecoxib_rediscovery', 'zaleplon_mpo', 'qed')."""

    def __init__(self, task_name: str):
        self.task_name = resolve_task_name(task_name)
        self.oracle = Oracle(name=self.task_name)

    def score(self, smiles: str) -> float:
        try:
            return float(self.oracle(smiles))
        except Exception:
            return 0.0
