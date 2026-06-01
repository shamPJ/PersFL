"""
Aggregate final-round accuracy (mean ± SD across seeds) for
CIFAR-10 rotated and sigma experiments. All algorithms — including
Ditto and Algorithm1_TopK (K=4, K=8) — appear in a single table
per experiment.

Usage:
    python aggregate_rotated.py
"""

import glob
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

BASE            = "results/cnn_cifar10_rotated"
BASE_SIGMA      = "results/cnn_cifar10_sigma"
BASE_TOPK       = "results/cnn_cifar10_topk_nclusters"
BASE_TOPK_SIGMA = "results/cnn_cifar10_topk_sigma"

ALGOS  = ["Algorithm1", "Ditto", "FedAvg", "FedBN", "FedProx", "IFCA"]
ALPHA  = 0.001

PATTERN            = re.compile(r"cnn_cifar10_c(\d+)_seed(\d+)\.csv")
PATTERN_SIGMA      = re.compile(r"cnn_cifar10_sigma(\d+)_seed(\d+)\.csv")
PATTERN_TOPK       = re.compile(r"cnn_cifar10_K(\d+)_nc(\d+)_seed(\d+)\.csv")
PATTERN_TOPK_SIGMA = re.compile(r"cnn_cifar10_K(\d+)_sigma(\d+)_seed(\d+)\.csv")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_algo(base, algo, pattern=PATTERN):
    """Return {key: [final_acc per seed]} for one algorithm subdirectory."""
    data = {}
    for f in sorted(glob.glob(os.path.join(base, algo, "*.csv"))):
        m = pattern.search(os.path.basename(f))
        if not m:
            continue
        key = int(m.group(1))
        df = pd.read_csv(f)
        data.setdefault(key, []).append(df["accuracy"].iloc[-1])
    return data


def load_topk(base, pattern):
    """Return {label: {key: [final_acc per seed]}} for TopK flat directory."""
    raw = {}
    for f in sorted(glob.glob(os.path.join(base, "*.csv"))):
        m = pattern.search(os.path.basename(f))
        if not m:
            continue
        K, key = int(m.group(1)), int(m.group(2))
        df = pd.read_csv(f)
        raw.setdefault(f"TopK (K={K})", {}).setdefault(key, []).append(
            df["accuracy"].iloc[-1]
        )
    return raw


def summarise(raw):
    """raw: {label: {key: [vals]}} → {label: {key: (mean, sd)}}"""
    out = {}
    for label, key_data in raw.items():
        out[label] = {}
        for key, vals in key_data.items():
            arr = np.array(vals)
            out[label][key] = (arr.mean(), arr.std(ddof=1))
    return out


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt(mean, sem):
    return f"{mean:.4f} ± {sem:.4f}"


# ── Statistics ────────────────────────────────────────────────────────────────

def holm_bonferroni(pvals, alpha=ALPHA):
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = np.array(pvals)[order]
    reject = np.zeros(n, dtype=bool)
    for i, p in enumerate(sorted_p):
        if p < alpha / (n - i):
            reject[order[i]] = True
        else:
            break
    return reject


def significance_tests(raw, keys, reference="Algorithm1", alpha=ALPHA):
    baselines = [a for a in raw if a != reference]
    comparisons = []
    for algo in baselines:
        for key in keys:
            ref_vals = np.array(raw[reference].get(key, []))
            alt_vals = np.array(raw[algo].get(key, []))
            if len(ref_vals) >= 2 and len(ref_vals) == len(alt_vals):
                t, p = stats.ttest_rel(ref_vals, alt_vals)
                comparisons.append((algo, key, t, p))
    if not comparisons:
        return {}
    pvals = [c[3] for c in comparisons]
    rejected = holm_bonferroni(pvals, alpha)
    return {(algo, key): (t, p, rej)
            for (algo, key, t, p), rej in zip(comparisons, rejected)}


# ── Printers ──────────────────────────────────────────────────────────────────

def print_table(rows, keys, col_label_fn, title, col_w=22):
    print(f"\n\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    header = f"{'Algorithm':<18}" + "".join(f"{col_label_fn(k):>{col_w}}" for k in keys)
    print(header)
    print("-" * len(header))
    for algo, kdata in rows.items():
        line = f"{algo:<18}"
        for k in keys:
            line += f"{fmt(*kdata[k]):>{col_w}}" if k in kdata else f"{'—':>{col_w}}"
        print(line)


def print_latex(rows, keys, comment):
    print(f"\n% --- LaTeX rows ({comment}) ---")
    for algo, kdata in rows.items():
        cells = " & ".join(
            f"${kdata[k][0]:.4f} \\pm {kdata[k][1]:.4f}$" if k in kdata else "—"
            for k in keys
        )
        print(f"        {algo:<18} & {cells} \\\\")


def print_significance_table(sig, baselines, keys, col_label_fn, reference="Algorithm1", alpha=ALPHA):
    print(f"\nPaired t-test vs {reference} (Holm–Bonferroni, α={alpha})")
    col_w = 20
    header = f"{'vs '+reference:<18}" + "".join(f"{col_label_fn(k):>{col_w}}" for k in keys)
    print(header)
    print("-" * len(header))
    for algo in baselines:
        line = f"{algo:<18}"
        for key in keys:
            if (algo, key) in sig:
                t, p, rej = sig[(algo, key)]
                marker = "*" if rej else " "
                line += f"{marker}p={p:.3f}(t={t:+.2f})".rjust(col_w)
            else:
                line += f"{'—':>{col_w}}"
        print(line)
    print(f"  (* significant at p < {alpha})")


def print_latex_significance(sig, baselines, keys, comment):
    print(f"\n% --- {comment} ---")
    for algo in baselines:
        cells = []
        for key in keys:
            if (algo, key) in sig:
                _, p, rej = sig[(algo, key)]
                cells.append(f"$p={p:.3f}$" + ("$^*$" if rej else ""))
            else:
                cells.append("—")
        print(f"        {algo:<18} & {' & '.join(cells)} \\\\")


def save_csv(rows, key_col, path):
    records = [
        {"algo": a, key_col: k, "mean": m, "sd": s}
        for a, kdata in rows.items()
        for k, (m, s) in kdata.items()
    ]
    pd.DataFrame(records).sort_values(["algo", key_col]).to_csv(
        path, index=False, float_format="%.6f"
    )
    print(f"\nSaved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CIFAR-10 rotated — one table
# ══════════════════════════════════════════════════════════════════════════════

raw_rot = {}
for algo in ALGOS:
    d = load_algo(BASE, algo)
    if d:
        raw_rot[algo] = d
raw_rot.update(load_topk(BASE_TOPK, PATTERN_TOPK))

rows_rot = summarise(raw_rot)
ks = sorted({k for d in rows_rot.values() for k in d})

print_table(rows_rot, ks, lambda k: f"k={k}", "CIFAR-10 rotated")
print_latex(rows_rot, ks, "CIFAR-10 rotated, mean ± SD per k")

REF = "TopK (K=4)"
baselines_rot = [a for a in rows_rot if a != REF]
sig_rot = significance_tests(raw_rot, ks, reference=REF)
print_significance_table(sig_rot, baselines_rot, ks, lambda k: f"k={k}", reference=REF)
print_latex_significance(sig_rot, baselines_rot, ks, "LaTeX significance — rotated")

save_csv(rows_rot, "n_clusters", "results/cnn_cifar10_rotated_summary.csv")

# ══════════════════════════════════════════════════════════════════════════════
# CIFAR-10 sigma — one table
# ══════════════════════════════════════════════════════════════════════════════

raw_sig = {}
for algo in ALGOS:
    d = load_algo(BASE_SIGMA, algo, pattern=PATTERN_SIGMA)
    if d:
        raw_sig[algo] = d
raw_sig.update(load_topk(BASE_TOPK_SIGMA, PATTERN_TOPK_SIGMA))

rows_sig = summarise(raw_sig)
sigmas = sorted({s for d in rows_sig.values() for s in d})

print_table(rows_sig, sigmas, lambda s: f"σ={s}", "CIFAR-10 sigma")
print_latex(rows_sig, sigmas, "CIFAR-10 sigma, mean ± SD per σ")

baselines_sig = [a for a in rows_sig if a != REF]
sig_sigma = significance_tests(raw_sig, sigmas, reference=REF)
print_significance_table(sig_sigma, baselines_sig, sigmas, lambda s: f"σ={s}", reference=REF)
print_latex_significance(sig_sigma, baselines_sig, sigmas, "LaTeX significance — sigma")

save_csv(rows_sig, "sigma", "results/cnn_cifar10_sigma_summary.csv")
