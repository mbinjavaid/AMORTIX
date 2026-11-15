# **AMORTIX: Amortized Molecular Optimization via Group Relative Policy Optimization**

AMORTIX is a framework for amortized molecular optimization. It adapts a pre-trained Graph Transformer model to optimize molecules via sequential atom-and-bond additions. To address the high variance arising from the heterogeneous difficulty of distinct starting structures, AMORTIX employs Group Relative Policy Optimization (GRPO) for goal-directed fine-tuning.

This approach normalizes rewards relative to the starting structure, stabilizing the learning process. The resulting policy generalizes to out-of-distribution molecular scaffolds and generates optimized molecules in a single forward pass without requiring inference-time oracle calls or iterative refinement.

## **Installation**

Install the dependencies found in requirements.txt (this includes PyTDC for the TDC objectives and wandb for optional logging):

`pip install -r requirements.txt`

Additionally, install **[torch_scatter](https://github.com/rusty1s/pytorch_scatter)** with options corresponding to your hardware.

Create the required directories:

`mkdir -p results data/chembl/pretrain_sequences`

## Pretraining

You can pretrain yourself or use the provided pretrained checkpoint.

### Pretrained checkpoint download

Download the pretrained checkpoint [here](https://drive.google.com/file/d/14HFad1ZQHlbU33J6D87uVWeNXcQRWAzl/view?usp=sharing).

You can use this checkpoint for fine-tuning on downstream tasks by specifying the path to the checkpoint in `config.py` (in variable `load_checkpoint_from_path`).


### Pretrain yourself

To pretrain with the settings in the paper, do the following: 

1. Download the file `chembl_35_chemreps.txt.gz` from this link: [Download File](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_chemreps.txt.gz)
and put the extracted `chembl_35_chemreps.txt` under the `./data/chembl` directory.
2. Run `$ python scripts/filter_molecules.py`. This will perform a train/val split with 100k validation molecules, which will be
put under `data/chembl/chembl_{train/valid}.smiles`. Then, depending on the allowed vocabulary (see `scripts/filter_molecules.py`),
it filters the smiles for all strings containing the allowed sets of characters and save them under `data/chembl/chembl_{train/valid}_filtered.smiles`.
3. Run `$ python scripts/create_pretrain_dataset.py`. This will convert the filtered SMILES to instances of `MoleculeDesign` (the class in `molecule_design.py`, which takes the role of the molecular graph environment).
The pickled molecules are saved in `./data/chembl/pretrain_sequences/chembl_{train/valid}.pickle`.
4. Run `$ python scripts/pretrain.py` to perform pretraining of the model. The general config to use (e.g., architecture) is under `config.py`.
In `scripts/pretrain.py`, you can specify pretrain-specific options directly at the entrypoint to adjust to your hardware, i.e.:
```python
if __name__ == '__main__':
    pretrain_train_dataset = "./data/chembl/pretrain_sequences/chembl_train.pickle"
    pretrain_val_dataset = "./data/chembl/pretrain_sequences/chembl_valid.pickle"
    pretrain_num_epochs = 1000   # For how many epochs to train
    batch_size = 512  # Minibatch size
    num_batches_per_epoch = 3000   # Number of minibatches per epoch.
    batch_size_validation = 512  # Batch size during validation
    training_device = "cuda:0"  # Device on which to train. Set to "cpu" if no CUDA available.
    num_dataloader_workers = 10  # Number of dataloader workers for creating batches for training
    load_checkpoint_from_path = None   # Path to checkpoint if training should be continued from existing weights.
```
5. The terminal output will show under which subdirectory (named after timestamp) in `./results` the script will save the model checkpoints.

6. After pretraining, the best model checkpoint (lowest validation loss) will be saved as `best_model.pt` in the corresponding subdirectory in `./results`. You can use this checkpoint for fine-tuning on downstream tasks by specifying the path to the checkpoint in `config.py` (in variable `load_checkpoint_from_path`).

## **Data Preparation**

Before running scaffold-constrained tasks, generate the cluster-based scaffold splits from the ZINC dataset to evaluate generalization to out-of-distribution scaffolds.

`cd scaffold_splitting && python smart_splitting.py`

The script reads `data/zinc/zinc.smiles` and writes `train_scaffolds.txt`, `val_scaffolds.txt` and `test_scaffolds.txt` for each split seed into `scaffold_splitting/zinc_splits_optimized/run_seed_<seed>/`. Point `fragment_library_path`, `validation_scaffolds_path` and `evaluation_scaffolds_path` in `config.py` to these files.

## **Usage and Configuration**

The framework is controlled via configuration files and command-line arguments. The primary configuration is located in config.py, where you can define the objective task, model hyperparameters, and enable reinforcement learning.

Key variables in `config.py`:

* `objective_type`: Sets the target objective (`"kinase_mpo"`, `"jnk3"`, `"prodrug_bbb"`, or any other task of the TDC oracle collection, e.g., `"drd2"`, `"celecoxib_rediscovery"`, `"zaleplon_mpo"`).
* `prodrug_bbb` scores a generated molecule against its parent as `P_BBB * QED gate * junction purity`: `P_BBB` is a MiniMol-based ensemble trained on TDC `bbb_martins` (trained on first use and cached in `oracle_cache/`), the QED gate requires QED >= `bbb_qed_floor`, and the purity is the fraction of bonds between parent and added part that are hydrolyzable.
* `max_oracle_calls`: Budget of unique molecules scored by the objective during training (default 50000). Training stops when it is used up; set to `None` for no limit.
* `use_dr_grpo`: Set to True to enable GRPO reinforcement learning. Set to False to run supervised fine-tuning.  
* `use_fragment_library`: Master switch for structural conditioning (scaffold-based generation).
* `use_grpo_grouping`: If True (GRPO), each starting structure is its own group with its own baseline. Set to False to pool all trajectories of an epoch into a single group with one global baseline, which gives REINFORCE. For plain REINFORCE also keep `rl_advantage_normalize = False` and `ppo_epochs = 1`.
* `use_wandb`: Set to True to log to Weights & Biases (off by default). Set `wandb_project` and `wandb_entity` accordingly.

### **Running an Experiment**

Run the training script using the default configuration:

`python main.py`

To use a different configuration module (e.g., a subclass of `MoleculeConfig` in its own file), use the \--config argument:

`python main.py --config my_config`

To evaluate a trained checkpoint without retraining:

`python scripts/evaluate_checkpoint.py --checkpoint results/<timestamp>/best_model.pt`

You can also override specific RL hyperparameters directly via the command line:

`python main.py --learning_rate 1e-4 --rl_entropy_beta 0.001 --ppo_epochs 1 --rl_ppo_clip_epsilon 0.2`

## **Output**

Results, model checkpoints, and logs are saved to the `results` directory by default:

* `best_model.pt` and `last_model.pt` weights.  
* CSV logs generated during evaluation (`<validation|test>_detailed_logs.csv`), containing starting SMILES, generated SMILES, and objective scores.
* Text files logging the top generated molecules per epoch (`epoch_<n>_train_top_20_molecules.txt`) and the best molecules found overall (`top10_all_time_smiles.txt`).
* `oracle_history.csv`: every unique molecule scored during training (canonical SMILES, in the order of the oracle calls) with its score. Duplicates are scored and counted only once.
* `pmo_top10_curve.csv`: the mean of the current top-10 scores each time it changed, together with the number of unique oracle calls so far.

If `use_wandb` is enabled, per-epoch training metrics (e.g., rewards, advantages, policy loss, number of unique oracle calls, top-10 mean and top-10 AUC over oracle calls) and validation scores are logged to the W&B project set by `wandb_project` / `wandb_entity` in `config.py`.
