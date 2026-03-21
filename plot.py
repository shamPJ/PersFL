import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

csvs = glob.glob(os.path.join("results/linear_syn_S", "linear_syn_S_*.csv"))

dfs = []
for f in csvs:
    fname = os.path.basename(f)
    core = fname.replace("linear_syn_S_", "").replace(".csv", "")
    S, SEED = core.split("_")

    tmp = pd.read_csv(f)
    tmp["seed"] = int(SEED)
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)

# -----------------------------
# Explicit metrics
# -----------------------------
metric_cols = ["loss_mean", "MSE_val", "MSE_params"]

S_vals = sorted(df['algo_S'].unique())
R = df['iter'].max() + 1
reps = df['seed'].nunique()

x = np.arange(R)
labels = [f"S={S}" for S in S_vals]

# -----------------------------
# Loop over metrics
# -----------------------------
for metric in metric_cols:
    mean_metric = []
    sem_metric = []

    for S in S_vals:
        df_S = df[df['algo_S'] == S]

        runs = np.zeros((reps, R))

        for r in range(R):
            vals = df_S[df_S['iter'] == r][metric].values
            runs[:, r] = vals

        mean_metric.append(np.mean(runs, axis=0))
        sem_metric.append(np.std(runs, axis=0) / np.sqrt(reps))

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(6, 4))

    for i in range(len(S_vals)):
        plt.plot(x, mean_metric[i], label=labels[i])
        plt.fill_between(
            x,
            mean_metric[i] - sem_metric[i],
            mean_metric[i] + sem_metric[i],
            alpha=0.3
        )

    plt.xlabel('Rounds R')
    plt.ylabel(metric)

    plt.yscale('log')

    plt.title(f'{metric} vs S')
    plt.legend(frameon=False)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{metric}_vs_S.png", dpi=300)
    plt.show()
    plt.close()