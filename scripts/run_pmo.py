"""
PMO-style benchmark runner (practical molecular optimization, 10,000 unique oracle calls, top-10 AUC).

The pre-trained model is fine-tuned with policy-gradient updates (see `rl_updates.dr_grpo_update`) while every molecule
is scored through one shared oracle (`utils.oracle_tracking.CentralOracle`) that de-duplicates by canonical SMILES, enforces the
oracle budget and tracks the top-10 AUC. Generation starts from the single carbon atom "C".

Each epoch draws `--samples_per_epoch` (128) IID samples from the policy. A random-forest surrogate trained on all
molecules scored so far picks the `--keep_per_epoch` (10) most promising unique ones (random ones until the surrogate
has at least 50 unique scored molecules); only these are evaluated with the oracle and used for the policy update.
`--no_surrogate` evaluates all samples instead.

Examples
--------
    python scripts/run_pmo.py --list_tasks
    python scripts/run_pmo.py                                             # all tasks, 3 seeded runs each
    python scripts/run_pmo.py --tasks drd2,jnk3,gsk3b --no_wandb --gpu 0  # several tasks, CSV logging only, GPU 0
    python scripts/run_pmo.py --task qed --runs 1                         # one task, one run
    python scripts/run_pmo.py --task qed --run_id 2                       # exactly run 2 of one task (seed 42 + 2)

Outputs (per run, in `<results_root>/<task>/run_<id>/`)
    oracle_history.csv     every unique oracle call: call_id, smiles, score
    pmo_top10_curve.csv    top-10 mean every time it changed: unique_calls, top10_avg, top10_smiles_scores
    pmo_epoch_log.csv      one row per epoch: unique calls, top-10 mean, top-10 AUC, best score, RL metrics
    pmo_summary.csv        final numbers of the run (also appended to `<results_root>/pmo_summary_all.csv`)
W&B (unless --no_wandb): the same per-epoch quantities, with `pmo_auc` being the top-10 AUC (flat extension of the
current top-10 mean up to the full budget).

REINFORCE / group notes
-----------------------
Every epoch uses a single start prompt ("C"), so there is exactly one group and the group-relative baseline equals
the batch-mean baseline, i.e. plain REINFORCE with a mean baseline. For multi-prompt training (`main.py` with
`use_fragment_library=True`) set `use_grpo_grouping=False` in `config.py` to pool all prompts into one group (global
baseline, REINFORCE) instead of one baseline per prompt (GRPO). Keep `rl_advantage_normalize=False` and
`ppo_epochs=1` for the plain REINFORCE update.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import argparse
import copy
import csv
import os
import subprocess
import sys
import time

# Tasks run by default (TDC oracle names).
PMO_TASKS = [
    "albuterol_similarity", "amlodipine_mpo", "celecoxib_rediscovery", "deco_hop", "drd2", "fexofenadine_mpo",
    "gsk3b", "isomers_c7h8n2o2", "isomers_c9h10n2o2pf2cl", "jnk3", "median1", "median2", "mestranol_similarity",
    "osimertinib_mpo", "perindopril_mpo", "qed", "ranolazine_mpo", "scaffold_hop", "sitagliptin_mpo",
    "thiothixene_rediscovery", "troglitazone_rediscovery", "zaleplon_mpo",
]

EPOCH_LOG_FIELDS = ["epoch", "unique_calls", "new_calls", "top10_avg", "pmo_auc", "best_score", "mean_reward",
                    "policy_loss", "mean_entropy", "rl_entropy_beta", "num_trajectories", "elapsed_s"]


def parse_args():
    p = argparse.ArgumentParser(description="PMO-style benchmark runner")
    p.add_argument("--task", help="Run only this task (TDC oracle name, e.g. qed, drd2, jnk3; see --list_tasks).")
    p.add_argument("--tasks", default="all",
                   help="Comma-separated task names or 'all' (default). Every task is run for --runs seeds.")
    p.add_argument("--runs", type=int, default=3, help="Number of seeded runs per task (default 3).")
    p.add_argument("--run_id", type=int, default=None,
                   help="Run only this single run (use together with --task); the seed is --seed_base + run_id.")
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--max_calls", type=int, default=10000, help="Oracle budget (unique molecules).")
    p.add_argument("--max_epochs", type=int, default=100000, help="Safety cap on the number of epochs.")
    p.add_argument("--samples_per_epoch", type=int, default=128,
                   help="IID samples drawn from the policy per epoch.")
    p.add_argument("--keep_per_epoch", type=int, default=10,
                   help="Number of the sampled molecules per epoch that are evaluated with the oracle (and used for "
                        "the update). They are selected by the surrogate model.")
    p.add_argument("--no_surrogate", action="store_true",
                   help="Disable surrogate filtering: all sampled molecules are evaluated with the oracle.")
    p.add_argument("--surrogate_n_jobs", type=int, default=-1, help="CPU threads of the surrogate random forest.")
    p.add_argument("--entropy", type=float, default=0.0, help="Entropy bonus coefficient (default 0).")
    p.add_argument("--lr", type=float, default=None, help="Learning rate. Default: value from config.py.")
    p.add_argument("--stall_threshold", type=int, default=None,
                   help="If fewer than this many new unique molecules were scored over the last 10 epochs, the "
                        "entropy bonus is raised by 0.001 (up to 0.01); if it is already at 0.01 the run stops. "
                        "Default: --keep_per_epoch.")
    p.add_argument("--no_stall_handling", action="store_true", help="Disable the stall detection described above.")
    p.add_argument("--checkpoint", default=None, help="Pre-trained checkpoint. Default: load_checkpoint_from_path of config.py")
    p.add_argument("--gpu", default=None, help="GPU id to use (sets CUDA_VISIBLE_DEVICES). Default: config.py value.")
    p.add_argument("--results_root", default="./results/pmo")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="amortix-pmo")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--list_tasks", action="store_true")
    return p.parse_args()


def launch_many(args):
    """Run every (task, run_id) as a separate process so that each run starts from a clean state."""
    if args.task:
        tasks = [args.task]
    elif args.tasks == "all":
        tasks = list(PMO_TASKS)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    passthrough = []
    for flag, val in [("--seed_base", args.seed_base), ("--max_calls", args.max_calls), ("--max_epochs", args.max_epochs),
                      ("--samples_per_epoch", args.samples_per_epoch), ("--keep_per_epoch", args.keep_per_epoch),
                      ("--surrogate_n_jobs", args.surrogate_n_jobs),
                      ("--entropy", args.entropy), ("--lr", args.lr), ("--stall_threshold", args.stall_threshold),
                      ("--checkpoint", args.checkpoint), ("--gpu", args.gpu), ("--results_root", args.results_root),
                      ("--wandb_project", args.wandb_project), ("--wandb_entity", args.wandb_entity)]:
        if val is not None:
            passthrough += [flag, str(val)]
    for flag, val in [("--no_stall_handling", args.no_stall_handling), ("--no_wandb", args.no_wandb),
                      ("--no_surrogate", args.no_surrogate)]:
        if val:
            passthrough.append(flag)

    for task in tasks:
        for run_id in range(1, args.runs + 1):
            print(f"\n{'=' * 60}\nTASK {task} | RUN {run_id}/{args.runs}\n{'=' * 60}\n", flush=True)
            cmd = [sys.executable, os.path.abspath(__file__), "--task", task, "--run_id", str(run_id)] + passthrough
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!! run failed (task={task}, run={run_id}): {e}", flush=True)
            except KeyboardInterrupt:
                print("\nInterrupted.")
                return


def run_single(args):
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"  # keep the GPU visible inside ray workers
    if args.gpu is not None:  # must happen before torch / ray are imported
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import random
    import ray
    import torch

    from config import MoleculeConfig
    from core.gumbeldore_dataset import GumbeldoreDataset
    from model.molecule_transformer import MoleculeTransformer
    from utils.oracle_tracking import CentralOracle
    from utils.chem_utils import SurrogateModel
    from rl_updates import dr_grpo_update

    task = args.task
    seed = args.seed_base + args.run_id
    entropy = args.entropy
    run_dir = os.path.join(args.results_root, task, f"run_{args.run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # ---- configuration -------------------------------------------------------------------------------------------
    config = MoleculeConfig()
    config.seed = seed
    config.objective_type = task
    config.results_path = run_dir
    config.log_to_file = False

    # Start every trajectory from the single carbon atom "C".
    config.use_fragment_library = False
    config.start_from_c_chains = True
    config.start_c_chain_max_len = 1
    config.repeat_start_instances = 1

    config.use_dr_grpo = True
    config.use_grpo_grouping = True  # one prompt per epoch -> one group (identical to REINFORCE with a mean baseline)
    config.rl_advantage_normalize = False
    config.ppo_epochs = 1
    config.rl_entropy_beta = entropy
    config.freeze_all_except_final_layer = False
    config.rl_use_novelty_bonus = False
    if args.lr is not None:
        config.optimizer["lr"] = args.lr
    if args.checkpoint is not None:
        config.load_checkpoint_from_path = args.checkpoint

    # Only the selected GPU is used: after CUDA_VISIBLE_DEVICES=<id> it appears as cuda:0 (the rollout workers
    # re-apply config.CUDA_VISIBLE_DEVICES, because ray hides the GPUs from tasks that do not request one).
    if args.gpu is not None:
        config.CUDA_VISIBLE_DEVICES = str(args.gpu)
        config.training_device = "cuda:0"
    else:
        first_gpu = config.CUDA_VISIBLE_DEVICES.split(",")[0] if config.CUDA_VISIBLE_DEVICES else "0"
        config.training_device = f"cuda:{first_gpu}"
    config.gumbeldore_config["devices_for_workers"] = [config.training_device]
    config.gumbeldore_config["search_type"] = "iid_mc"
    config.gumbeldore_config["num_samples_per_instance"] = args.samples_per_epoch
    config.gumbeldore_config["num_trajectories_to_keep"] = args.keep_per_epoch
    config.gumbeldore_config["destination_path"] = None

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ---- logging -------------------------------------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=f"{task}_run{args.run_id}_seed{seed}",
                   config=dict(task=task, run_id=args.run_id, seed=seed, max_calls=args.max_calls, entropy=entropy,
                               lr=config.optimizer["lr"], search_type="iid_mc",
                               samples_per_epoch=args.samples_per_epoch, keep_per_epoch=args.keep_per_epoch,
                               surrogate=not args.no_surrogate, stall_threshold=args.stall_threshold,
                               stall_handling=not args.no_stall_handling))

    epoch_log_path = os.path.join(run_dir, "pmo_epoch_log.csv")
    with open(epoch_log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=EPOCH_LOG_FIELDS).writeheader()

    # ---- ray / oracle / model ------------------------------------------------------------------------------------
    if ray.is_initialized():
        ray.shutdown()
    num_gpus = len(config.CUDA_VISIBLE_DEVICES.split(",")) if config.CUDA_VISIBLE_DEVICES else 1
    ray.init(num_gpus=num_gpus, logging_level="info", ignore_reinit_error=True)

    oracle = CentralOracle.remote(task, run_dir, args.max_calls)
    ray.get(oracle.get_budget_status.remote())  # wait until the TDC oracle is loaded

    network = MoleculeTransformer(config, config.training_device)
    if config.load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {config.load_checkpoint_from_path}")
        checkpoint = torch.load(config.load_checkpoint_from_path)
        if checkpoint.get("model_weights") is not None:
            network.load_state_dict(checkpoint["model_weights"])
    network.to(network.device)
    network.eval()

    optimizer = torch.optim.Adam(network.parameters(), lr=config.optimizer["lr"],
                                 weight_decay=config.optimizer["weight_decay"])
    dataset = GumbeldoreDataset(config=config, objective_evaluator=None, central_oracle=oracle)

    # Surrogate: a random forest on the molecules scored so far. Of the samples drawn each epoch only `keep_per_epoch`
    # are evaluated with the oracle: the ones the surrogate ranks highest (random ones until it has enough data).
    surrogate = None if args.no_surrogate else SurrogateModel(n_jobs=args.surrogate_n_jobs)
    surrogate_buffer = {"smiles": [], "scores": []}
    stall_threshold = args.stall_threshold if args.stall_threshold is not None else args.keep_per_epoch

    # ---- training loop -------------------------------------------------------------------------------------------
    start_time = time.time()
    new_calls_history = []
    stats = ray.get(oracle.get_stats.remote())
    stop_reason = "max_epochs"
    epoch = 0

    for epoch in range(1, args.max_epochs + 1):
        if stats["unique_calls"] >= args.max_calls:
            stop_reason = "budget_reached"
            break

        if not args.no_stall_handling and len(new_calls_history) >= 10:
            if sum(new_calls_history[-10:]) < stall_threshold:
                if config.rl_entropy_beta < 0.01:
                    config.rl_entropy_beta += 0.001
                    print(f">> Stalling detected. Entropy beta raised to {config.rl_entropy_beta:.4f}")
                    new_calls_history = []
                else:
                    print(">> Stalling despite maximal entropy bonus. Stopping.")
                    stop_reason = "stalled"
                    break

        calls_before = stats["unique_calls"]
        network_weights = copy.deepcopy(network.get_weights())
        network.eval()
        trajectories = dataset.generate_dataset(network_weights=network_weights, best_objective=None,
                                                memory_aggressive=False, mode="train",
                                                surrogate_filter=None if surrogate is None else
                                                {"model": surrogate, "num_keep": args.keep_per_epoch})

        metrics = {}
        if trajectories and any(trajectories):
            network.train()
            metrics = dr_grpo_update(model=network, optimizer=optimizer, designs_groups=trajectories, config=config,
                                     device=torch.device(config.training_device), logger=None, novelty_memory=None)
            network.eval()
        else:
            print("WARNING: no trajectories generated this epoch, skipping update.")

        if surrogate is not None:
            for group in trajectories or []:
                for m in group:
                    if m.objective is not None and m.objective > float("-inf") and m.smiles_string:
                        surrogate_buffer["smiles"].append(m.smiles_string)
                        surrogate_buffer["scores"].append(float(m.objective))
            surrogate.train(surrogate_buffer["smiles"], surrogate_buffer["scores"])

        stats = ray.get(oracle.get_stats.remote())
        new_calls = stats["unique_calls"] - calls_before
        new_calls_history.append(new_calls)

        row = {
            "epoch": epoch,
            "unique_calls": stats["unique_calls"],
            "new_calls": new_calls,
            "top10_avg": stats["top10_avg"],
            "pmo_auc": stats["pmo_auc"],
            "best_score": stats["best_score"],
            "mean_reward": metrics.get("mean_reward", float("nan")),
            "policy_loss": metrics.get("policy_loss", float("nan")),
            "mean_entropy": metrics.get("mean_entropy", float("nan")),
            "rl_entropy_beta": config.rl_entropy_beta,
            "num_trajectories": metrics.get("num_trajectories", 0),
            "elapsed_s": round(time.time() - start_time, 1),
        }
        with open(epoch_log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=EPOCH_LOG_FIELDS).writerow(row)
        if use_wandb:
            wandb.log({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        print(f">> epoch {epoch} | calls {stats['unique_calls']}/{args.max_calls} | top10 {stats['top10_avg']:.4f} | "
              f"AUC {stats['pmo_auc']:.4f} | best {stats['best_score']:.4f}", flush=True)

    # ---- final summary -------------------------------------------------------------------------------------------
    stats = ray.get(oracle.get_stats.remote())
    summary = {
        "task": task, "run_id": args.run_id, "seed": seed, "epochs": epoch,
        "unique_calls": stats["unique_calls"], "max_calls": args.max_calls,
        "top10_auc": stats["pmo_auc"], "top10_avg_final": stats["top10_avg"], "best_score": stats["best_score"],
        "stop_reason": stop_reason, "wall_clock_s": round(time.time() - start_time, 1),
    }
    with open(os.path.join(run_dir, "pmo_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)
    all_path = os.path.join(args.results_root, "pmo_summary_all.csv")
    write_header = not os.path.exists(all_path)
    with open(all_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if write_header:
            w.writeheader()
        w.writerow(summary)

    print(f">> Final: task={task} run={args.run_id} calls={stats['unique_calls']} top10_auc={stats['pmo_auc']:.4f} "
          f"(stop: {stop_reason})")
    if use_wandb:
        wandb.summary.update({"top10_auc": stats["pmo_auc"], "top10_avg_final": stats["top10_avg"],
                              "best_score": stats["best_score"], "unique_calls": stats["unique_calls"]})
        wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    if args.list_tasks:
        print("\n".join(PMO_TASKS))
    elif args.task and args.run_id is not None:
        run_single(args)
    else:
        launch_many(args)
