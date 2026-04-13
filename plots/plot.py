import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dir = "cnn_cifar10_iid/Algorithm1"
# dir = "cnn_cifar10_iid/FedAvg"
csvs = glob.glob(os.path.join("results", dir, "cnn_cifar10_iid_0.csv"))
print(csvs)
# dfs = []
# for f in csvs:
    # fname = os.path.basename(f)
    # core = fname.replace("linear_syn_dm_", "").replace(".csv", "")
    # PARAM, SEED = core.split("_")

    # tmp = pd.read_csv(f)
    # tmp["seed"] = int(SEED)
    # dfs.append(tmp)
    # df = pd.read_csv(f)

# df = pd.concat(dfs, ignore_index=True)
df = pd.read_csv(csvs[0], sep=',')
# df = pd.read_csv(csvs, sep=',')
plt.plot(df['iter'], df['loss_mean'])
plt.show()
plt.plot(df['iter'], df['accuracy'])
plt.show()
# # -----------------------------
# # Explicit metrics
# # -----------------------------
# metric_cols = ["loss_mean", "MSE_val", "MSE_params"]

# param_vals = sorted(df['data_n_features'].unique())
# R = df['iter'].max() + 1
# reps = df['seed'].nunique()

# x = np.arange(R)
# labels = [f"dm={param}" for param in param_vals]

# # -----------------------------
# # Loop over metrics
# # -----------------------------
# for metric in metric_cols:
#     mean_metric = []
#     sem_metric = []

#     for param in param_vals:
#         df_p = df[df['data_n_features'] == param]

#         runs = np.zeros((reps, R))

#         for r in range(R):
#             vals = df_p[df_p['iter'] == r][metric].values
#             runs[:, r] = vals

#         mean_metric.append(np.mean(runs, axis=0))
#         sem_metric.append(np.std(runs, axis=0) / np.sqrt(reps))

#     # -----------------------------
#     # Plot
#     # -----------------------------
#     plt.figure(figsize=(6, 4))

#     for i in range(len(param_vals)):
#         plt.plot(x, mean_metric[i], label=labels[i])
#         plt.fill_between(
#             x,
#             mean_metric[i] - sem_metric[i],
#             mean_metric[i] + sem_metric[i],
#             alpha=0.3
#         )

#     plt.xlabel('Rounds R')
#     plt.ylabel(metric)

#     plt.yscale('log')

#     plt.title(f'{metric} vs S')
#     plt.legend(frameon=False)
#     plt.grid(True)
#     plt.tight_layout()

#     plt.savefig(f"{metric}_vs_dm_algo2.png", dpi=300)
#     plt.show()
#     plt.close()