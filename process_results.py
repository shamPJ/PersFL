import numpy as np
import pandas as pd
import glob
import os
import re

EXPERIMENT_META = {
    "linear_syn_dm": "data_n_features",
    # "linear_syn_noise": "data_noise_scale",
    # "linear_syn_noise_w": "data_noise_weight",
    # "linear_syn_nclusters": "data_n_clusters",
    # "linear_syn_S": "algo_S",
    # "linear_syn_R": "algo_R_local",
    # "linear_syn_lmbd": "algo_lmbd"
}

ALGO_METRIC = {
    "Algorithm2_SKLearn": "MSE_val",
    "Algorithm2_SKLearn_local": "MSE_val"
}

def aggregate_for_pgfplots(
    input_dir,
    pattern,
    param_name="param",
    output_file="aggregated.csv",
    metric_name="MSE_params",   
):
    files = glob.glob(os.path.join(input_dir, "*.csv"))

    data = {}
    
    for f in files:
        fname = os.path.basename(f)
        match = re.search(pattern, fname)
        if not match:
            continue

        param = float(match.group(1))
        seed = int(match.group(2))

        df = pd.read_csv(f)

        # extract chosen metric
        y = df[metric_name].values

        if param not in data:
            data[param] = []

        data[param].append(y)

    params_sorted = sorted(data.keys())

    # iteration column
    iters = df["iter"].values
    out = pd.DataFrame({"iter": iters})

    for p in params_sorted:
        runs = data[p]

        # ensure equal length
        min_len = min(len(r) for r in runs)
        runs = np.stack([r[:min_len] for r in runs])

        mean = runs.mean(axis=0)
        sem = runs.std(axis=0) / np.sqrt(runs.shape[0])

        out[f"{param_name}_{p}_mean"] = mean
        out[f"{param_name}_{p}_upper"] = mean + sem
        out[f"{param_name}_{p}_lower"] = mean - sem
    out.to_csv(output_file, index=False)
    print(f"Saved → {output_file}")

def make_pattern(exp_name):
    r"""
    Regex pattern:
    rf"{exp_name}_(\d+(?:\.\d+)?)_(\d+)\.csv"
    
    \d        → any digit (0-9)
    \d+       → one or more digits (e.g., 3, 42, 100)
    
    (\d+(?:\.\d+)?) → captures a number:
      - \d+          → integer part
      - (?:\.\d+)?   → optional decimal part (e.g., ".5")
      → matches both integers (10) and floats (10.5)
    
    (?:\.\d+)?  → optional decimal part
    (...)   → grouping
    ?:      → non-capturing group (group, but don't store it separately)
    \.      → literal dot "."
    \d+     → one or more digits
    (...)? → the whole group is optional

    (\d+) → captures the seed (integer)
    
    \.csv → matches the literal ".csv" (dot must be escaped)
    
    rf""  → raw f-string:
      - r = raw string (no escaping of backslashes)
      - f = allows inserting {exp_name}
    """
    return rf"{exp_name}_(\d+(?:\.\d+)?)_(\d+)\.csv"

BASE_DIR = "results"
OUTPUT_DIR = "results/aggregated_csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for exp_name in os.listdir(BASE_DIR):
    exp_path = os.path.join(BASE_DIR, exp_name)

    if not os.path.isdir(exp_path):
        continue

    if exp_name not in EXPERIMENT_META:
        continue  # skip irrelevant folders

    param_name = EXPERIMENT_META[exp_name]

    # find algorithms inside experiment
    for algo in os.listdir(exp_path):
        algo_path = os.path.join(exp_path, algo)

        if not os.path.isdir(algo_path):
            continue

        pattern = make_pattern(exp_name)

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{exp_name}_{algo}.csv"
        )

        print(f"Processing: {exp_name} / {algo}")

        metric_name = ALGO_METRIC.get(algo, "MSE_params")

        aggregate_for_pgfplots(
            input_dir=algo_path,
            pattern=pattern,
            param_name=param_name,
            output_file=output_file,
            metric_name=metric_name,
        )

