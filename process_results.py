import numpy as np
import pandas as pd
import glob
import os
import re

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

if __name__ == "__main__":
    # aggregate_for_pgfplots(
    #     input_dir="results/linear_syn_dm",
    #     pattern=r"linear_syn_dm_(\d+(?:\.\d+)?)_(\d+)\.csv",
    #     param_name="data_n_features",
    #     output_file="aggregated_dm.csv",
    #     metric_name="MSE_params",   
    # )

    aggregate_for_pgfplots(
        input_dir="results/linear_syn_dm/Algorithm2",
        pattern=r"linear_syn_dm_(\d+(?:\.\d+)?)_(\d+)\.csv",
        param_name="data_n_features",
        output_file="aggregated_dm_Algorithm2.csv",
        metric_name="MSE_params",   
    )

    # aggregate_for_pgfplots(
    #     input_dir="results/linear_syn_noise",
    #     pattern=r"linear_syn_noise_(\d+(?:\.\d+)?)_(\d+)\.csv",
    #     param_name="data_noise_scale",
    #     output_file="aggregated_noise.csv",
    #     metric_name="MSE_params",   
    # )

    # aggregate_for_pgfplots(
    #     input_dir="results/linear_syn_S",
    #     pattern=r"linear_syn_S_(\d+)_(\d+)\.csv",
    #     param_name="algo_S",
    #     output_file="aggregated_S.csv",
    #     metric_name="MSE_params",   
    # )

    # aggregate_for_pgfplots(
    #     input_dir="results/linear_syn_weight_noise",
    #     pattern=r"linear_syn_weight_noise_(\d+(?:\.\d+)?)_(\d+)\.csv",
    #     param_name="data_noise_weight",
    #     output_file="aggregated_noise_weight.csv",
    #     metric_name="MSE_params",   
    # )

    # aggregate_for_pgfplots(
    #     input_dir="results/linear_syn_nclusters",
    #     pattern=r"linear_syn_nclusters_(\d+(?:\.\d+)?)_(\d+)\.csv",
    #     param_name="data_n_clusters",
    #     output_file="aggregated_nclusters.csv",
    #     metric_name="MSE_params",   
    # )

    # aggregate_for_pgfplots(
    #     input_dir="results/linear_syn_R",
    #     pattern=r"linear_syn_R_(\d+(?:\.\d+)?)_(\d+)\.csv",
    #     param_name="algo_R_local",
    #     output_file="aggregated_R.csv",
    #     metric_name="MSE_params",   
    # )