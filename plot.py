import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# Load CSV (optional if you already have `rows` list)
# df = pd.read_csv("experiment_results.csv")

import glob
import os
import pandas as pd

csvs = glob.glob(os.path.join("results/linear_syn_dm", "linear_syn_dm_*.csv"))

dfs = []
for f in csvs:
    # extract filename
    fname = os.path.basename(f)
    
    # remove prefix and suffix
    core = fname.replace("linear_syn_dm_", "").replace(".csv", "")
    # split into parts
    D, SEED = core.split("_")
    # read file
    tmp = pd.read_csv(f)
    # add variables
    tmp["seed"] = int(SEED)
    
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)

dims = sorted(df['data_n_features'].unique())  # e.g., [2, 10]
R = df['iter'].max() + 1                      # number of rounds
reps = df['seed'].max() + 1                     # number of repetitions

mean_mse = []
sem_mse = []

for d in dims:
    # Filter rows with this n_features
    df_d = df[df['data_n_features'] == d]

    # Create array of shape (reps, R)
    runs = np.zeros((reps, R))
    for r in range(R):
        # mean MSE across repetitions for this round
        runs[:, r] = df_d[df_d['iter'] == r]['mse'].values 

    mean_mse.append(np.mean(runs, axis=0))
    sem_mse.append(np.std(runs, axis=0) / np.sqrt(runs.shape[0]))

    labels = [f"d={d}" for d in dims]  # can customize as d/m ratio
x = np.arange(R)

plt.figure(figsize=(6, 4))

for i in range(len(dims)):
    plt.plot(x, mean_mse[i], label=labels[i])
    plt.fill_between(
        x,
        mean_mse[i] - sem_mse[i],
        mean_mse[i] + sem_mse[i],
        alpha=0.3
    )

plt.xlabel('Rounds R')
plt.ylabel('MSE')
plt.yscale('log')
plt.title('PersFL MSE vs n_features')
plt.legend(frameon=False)
plt.grid(True)
plt.tight_layout()
plt.savefig("MSE_vs_nfeatures_updated.png", dpi=300)
plt.show()
plt.close()
