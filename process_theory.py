import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

RESULTS_DIR = "results/linear_syn_theory/Algorithm1"
FIG_DIR     = "plots"
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------
# Load all _theory.csv files
# filename pattern: linear_syn_{M}_{SEED}_theory.csv
# -----------------------------------------------
files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_theory.csv")))
if not files:
    raise FileNotFoundError(f"No theory CSV files found in {RESULTS_DIR}")

dfs = []
for f in files:
    match = re.search(r"linear_syn_(\d+)_(\d+)_theory\.csv", os.path.basename(f))
    if not match:
        continue
    m, seed = int(match.group(1)), int(match.group(2))
    df = pd.read_csv(f)
    df["m"]    = m
    df["seed"] = seed
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# -----------------------------------------------
# Derived quantities
# -----------------------------------------------
# delta: gap of each candidate vs best candidate in same (seed, m, client, round)
min_true = data.groupby(["seed", "m", "client", "round"])["true_loss_cand"].transform("min")
data["delta"] = data["true_loss_cand"] - min_true

# selected candidate: argmin empirical loss per (seed, m, client, round)
data["est_loss"] = data["true_loss_cand"] + data["xi"]
sel_idx = data.groupby(["seed", "m", "client", "round"])["est_loss"].idxmin()
data["is_selected"] = False
data.loc[sel_idx, "is_selected"] = True

# squared quantities for variance
data["xi_sq"]         = data["xi"] ** 2
data["true_loss_sq"]  = data["true_loss_cand"] ** 2

# -----------------------------------------------
# Aggregate by (m, round) — mean across seeds, clients, candidates
# -----------------------------------------------
def agg_round(df):
    within  = df[df["candidate_same_cluster"]]
    between = df[~df["candidate_same_cluster"]]
    sel     = df[df["is_selected"]]

    return pd.Series({
        # noise
        "var_emp":              df["xi_sq"].mean(),
        "true_loss_sq_mean":    df["true_loss_sq"].mean(),
        # gap
        "delta_within_mean":    within["delta"].mean()  if len(within)  else np.nan,
        "delta_between_mean":   between["delta"].mean() if len(between) else np.nan,
        "delta_between_min":    between["delta"].min()  if len(between) else np.nan,
        # selection quality
        "same_cluster_rate":    sel["candidate_same_cluster"].mean() if len(sel) else np.nan,
    })

stats = (
    data.groupby(["m", "round"])
    .apply(agg_round)
    .reset_index()
)

stats["var_theory"] = (2 / stats["m"]) * stats["true_loss_sq_mean"]
stats["sigma"]      = np.sqrt(stats["var_theory"])
stats["SNR"]        = stats["delta_between_min"] / stats["sigma"]

M_VALUES = sorted(stats["m"].unique())
colors   = plt.cm.tab10(np.linspace(0, 0.5, len(M_VALUES)))

# -----------------------------------------------
# Plot
# -----------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax1, ax2, ax3, ax4 = axes.ravel()

for m, c in zip(M_VALUES, colors):
    s = stats[stats["m"] == m].sort_values("round")
    label = f"m={m}"

    # 1 — noise: analytical vs empirical
    ax1.plot(s["round"], s["var_emp"],    color=c, ls="-",  label=f"{label} emp")
    ax1.plot(s["round"], s["var_theory"], color=c, ls="--", label=f"{label} theory")

    # 2 — SNR
    ax2.plot(s["round"], s["SNR"], color=c, label=label)

    # 3 — candidate selection quality
    ax3.plot(s["round"], s["same_cluster_rate"], color=c, label=label)

    # 4 — gap within vs between
    ax4.plot(s["round"], s["delta_within_mean"],  color=c, ls="-",  label=f"{label} within")
    ax4.plot(s["round"], s["delta_between_mean"], color=c, ls="--", label=f"{label} between")

ax1.set_title(r"Noise: $\mathrm{Var}(\xi)$ — empirical vs $(2/m)\,\mathbb{E}[L_1^2]$")
ax1.set_xlabel("round")
ax1.set_ylabel(r"$\mathrm{Var}(\xi)$")
ax1.legend(fontsize=7)

ax2.set_title(r"SNR $= \Delta_{\min}^{\mathrm{between}} / \sigma$")
ax2.set_xlabel("round")
ax2.set_ylabel("SNR")
ax2.axhline(0, color="k", lw=0.5, ls=":")
ax2.legend()

ax3.set_title("Candidate selection quality — same-cluster rate")
ax3.set_xlabel("round")
ax3.set_ylabel("fraction selected from same cluster")
ax3.set_ylim(0, 1)
ax3.legend()

ax4.set_title(r"Gap: within-cluster vs between-cluster $\Delta$")
ax4.set_xlabel("round")
ax4.set_ylabel(r"mean $\Delta$")
ax4.legend(fontsize=7)

fig.tight_layout()
out = os.path.join(FIG_DIR, "theory_analysis.png")
fig.savefig(out, dpi=150)
print(f"Saved {out}")
