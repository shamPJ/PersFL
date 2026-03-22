import argparse
import yaml
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--slurm_array_id", type=int, default=0)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

experiments = cfg["experiments"]
seeds = cfg["SEEDS"]

# Build a full list of experiments repeated by seeds
full_experiments = []
for exp in experiments:
    for seed in seeds:
        full_experiments.append({
            "d": exp["D"],
            "n_clusters": exp["N_CLUSTERS"],
            "seed": seed
        })

# Pick experiment by SLURM array ID
try:
    selected = full_experiments[args.slurm_array_id]
except IndexError:
    raise ValueError(f"SLURM array ID {args.slurm_array_id} out of range (0-{len(full_experiments)-1})")

D = selected["d"]
N_CLUSTERS = selected["n_clusters"]
SEED = selected["seed"]

OUT_DIR = cfg.get("OUT_DIR", "results/experiments")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = cfg.get("DEVICE", "cpu")

FNAME = os.path.join(OUT_DIR, f"linear_syn_D{D}_C{N_CLUSTERS}_S{SEED}.csv")

cmd = [
    "python", "main.py",
    "--n_clients", str(cfg["n_clients"]),
    "--n_clusters", str(N_CLUSTERS),
    "--n_features", str(D),
    "--model", cfg["model"],
    "--dataset", cfg["dataset"],
    "--algo", cfg["algo"],
    "--R", str(cfg["R"]),
    "--lrate", str(cfg["lrate"]),
    "--S", str(cfg["S"]),
    "--fname", FNAME,
    "--device", DEVICE,
    "--problem", cfg["problem"],
    "--seed", str(SEED)
]

print(f"Running experiment: D={D}, N_CLUSTERS={N_CLUSTERS}, SEED={SEED}")
subprocess.run(cmd, check=True)