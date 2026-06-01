"""
Compute mean ± SD of iterations needed to first reach THRESHOLD accuracy,
for results/cnn_cifar10_rotated and results/cnn_cifar10_sigma.
Prints console tables, LaTeX rows, and saves summary CSVs.

Usage:
    python aggregate_iters_to_threshold.py
"""

import glob
import os
import re
import numpy as np
import pandas as pd

THRESHOLD = 0.40

_HERE          = os.path.dirname(os.path.abspath(__file__))
BASE           = os.path.join(_HERE, "results/cnn_cifar10_rotated")
BASE_SIGMA     = os.path.join(_HERE, "results/cnn_cifar10_sigma")
BASE_TOPK      = os.path.join(_HERE, "results/cnn_cifar10_topk_nclusters")
BASE_TOPK_SIGMA = os.path.join(_HERE, "results/cnn_cifar10_topk_sigma")
ALGOS          = ["Algorithm1", "FedAvg", "FedBN", "FedProx", "IFCA", "Ditto"]
PATTERN        = re.compile(r"cnn_cifar10_c(\d+)_seed(\d+)\.csv")
PATTERN_SIGMA  = re.compile(r"cnn_cifar10_sigma(\d+)_seed(\d+)\.csv")
PATTERN_TOPK   = re.compile(r"cnn_cifar10_K(\d+)_nc(\d+)_seed(\d+)\.csv")
PATTERN_TOPK_SIGMA = re.compile(r"cnn_cifar10_K(\d+)_sigma(\d+)_seed(\d+)\.csv")


def iters_to_threshold_topk(base_dir, pattern, threshold=THRESHOLD):
    """Return {label: {key: [first iter >= threshold]}} for a flat TopK directory."""
    data = {}
    for f in sorted(glob.glob(os.path.join(base_dir, "*.csv"))):
        m = pattern.search(os.path.basename(f))
        if not m:
            continue
        K, key = int(m.group(1)), int(m.group(2))
        label = f"TopK (K={K})"
        df = pd.read_csv(f)
        hit = df[df["accuracy"] >= threshold]["iter"]
        iters = float(hit.iloc[0]) if len(hit) else float("nan")
        data.setdefault(label, {}).setdefault(key, []).append(iters)
    return data


def iters_to_threshold(algo_dir, pattern, threshold=THRESHOLD):
    """Return {key: [first iter where accuracy >= threshold, per seed]}."""
    data = {}
    for f in sorted(glob.glob(os.path.join(algo_dir, "*.csv"))):
        m = pattern.search(os.path.basename(f))
        if not m:
            continue
        key = int(m.group(1))
        df = pd.read_csv(f)
        hit = df[df["accuracy"] >= threshold]["iter"]
        iters = float(hit.iloc[0]) if len(hit) else float("nan")
        data.setdefault(key, []).append(iters)
    return data


def aggregate(data_by_key):
    """Return {key: (mean, sd)} from {key: [values]}."""
    result = {}
    for key, vals in sorted(data_by_key.items()):
        arr = np.array(vals)
        mean = np.nanmean(arr)
        sd   = np.nanstd(arr, ddof=1)
        result[key] = (mean, sd)
    return result


def fmt(mean, sem):
    return f"{mean:.1f} ± {sem:.2f}"


def print_table(rows, keys, col_label_fn, title):
    print(f"\n=== {title} ===")
    header = f"{'Algorithm':<14}" + "".join(f"{col_label_fn(k):>20}" for k in keys)
    print(header)
    print("-" * len(header))
    for algo, kdata in rows.items():
        line = f"{algo:<14}"
        for k in keys:
            line += f"{fmt(*kdata[k]):>20}" if k in kdata else f"{'—':>20}"
        print(line)


def print_latex(rows, keys, comment):
    print(f"\n% --- {comment} ---")
    for algo, kdata in rows.items():
        cells = " & ".join(
            f"${kdata[k][0]:.1f} \\pm {kdata[k][1]:.2f}$" if k in kdata else "—"
            for k in keys
        )
        print(f"        {algo:<14} & {cells} \\\\")


def save_csv(rows, keys, key_col, path):
    records = [
        {"algo": algo, key_col: k, "mean": mean, "sd": sd}
        for algo, kdata in rows.items()
        for k, (mean, sd) in kdata.items()
    ]
    if not records:
        print(f"No data to save → {path}")
        return
    pd.DataFrame(records).sort_values(["algo", key_col]).to_csv(
        path, index=False, float_format="%.4f"
    )
    print(f"Saved → {path}")


# ── Rotated (vary k) ───────────────────────────────────────────────────────────
rows_rot = {}
for algo in ALGOS:
    algo_dir = os.path.join(BASE, algo)
    if not os.path.isdir(algo_dir):
        continue
    rows_rot[algo] = aggregate(iters_to_threshold(algo_dir, PATTERN))

for label, kdata in iters_to_threshold_topk(BASE_TOPK, PATTERN_TOPK).items():
    rows_rot[label] = aggregate(kdata)

ks = sorted({k for d in rows_rot.values() for k in d})
print_table(rows_rot, ks, lambda k: f"k={k}", f"Iters to {THRESHOLD:.0%} acc (rotated)")
print_latex(rows_rot, ks, f"LaTeX rows — rotated, iters to reach {THRESHOLD:.0%} per k")
save_csv(rows_rot, ks, "n_clusters", os.path.join(_HERE, "results/cnn_cifar10_rotated_iters_summary.csv"))

# ── Sigma ──────────────────────────────────────────────────────────────────────
rows_sig = {}
for algo in ALGOS:
    algo_dir = os.path.join(BASE_SIGMA, algo)
    if not os.path.isdir(algo_dir):
        continue
    rows_sig[algo] = aggregate(iters_to_threshold(algo_dir, PATTERN_SIGMA))

for label, kdata in iters_to_threshold_topk(BASE_TOPK_SIGMA, PATTERN_TOPK_SIGMA).items():
    rows_sig[label] = aggregate(kdata)

sigmas = sorted({s for d in rows_sig.values() for s in d})
print_table(rows_sig, sigmas, lambda s: f"σ={s}", f"Iters to {THRESHOLD:.0%} acc (sigma)")
print_latex(rows_sig, sigmas, f"LaTeX rows — sigma, iters to reach {THRESHOLD:.0%} per σ")
save_csv(rows_sig, sigmas, "sigma", os.path.join(_HERE, "results/cnn_cifar10_sigma_iters_summary.csv"))
