from registry import MODELS, DATASETS, ALGOS
from options import read_options
from itertools import product
import pandas as pd
import importlib
import torch
import random
import numpy as np
import os

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_object(module, name):
    # e.g. for model/linreg.py with udf LinReg() - module = 'model.linreg' and 'name'='LinReg'
    mod = importlib.import_module(module)
    return getattr(mod, name)

def experiment_combinations(grid, repetitions):
    keys, values = zip(*grid.items())
    combos = list(product(*values))
    for rep in range(repetitions):
        seed_everything(rep) # diff seed for each rep
        for combo in combos:
            yield rep, dict(zip(keys, combo))

def merge_defaults(defaults, overrides):
    """
    Merge two dictionaries:
    - overwrite defaults with values from overrides **only for existing keys**
    - ignore any keys in overrides that are not in defaults
    """
    # get(key[, default]) - Return the value for key if key is in the dictionary, else default.
    # value is returned if key doesnt exist
    return {k: overrides.get(k, v) for k, v in defaults.items()}

if __name__ == "__main__":
    
    parsed = read_options()

    model_spec = MODELS[parsed["model"]] # choose dataset, model, algo from user input or default vals
    data_spec = DATASETS[parsed["dataset"]] # e.g. 'linreg', 'synthetic', 'persfl'; corresponds to .py files
    algo_spec = ALGOS[parsed["algo"]]
    print(model_spec.module, model_spec.cls)
    ModelCls = load_object(model_spec.module, model_spec.cls)
    AlgoCls = load_object(algo_spec.module, algo_spec.cls)
    DataCls = load_object(data_spec.module, data_spec.loader) # data gen function or pytorch data loader

    # Example parameter grid for sweep
    GRID = {
        "n_features": [2, 10, 50, 100],
        "n_samples": [10],
        "lrate": [0.03]
    }

    rows = []
    for rep, grid_params in experiment_combinations(GRID, parsed["reps"]):
        print("rep", rep)
        # Merge grid with defaults (overwrite default vals if passed)
        # defaults <- grid
        tmp_data_params  = merge_defaults(data_spec.default_params, grid_params)
        tmp_model_params = merge_defaults(model_spec.default_params, grid_params)
        tmp_algo_params  = merge_defaults(algo_spec.default_params, grid_params)
        print(tmp_data_params)
        # (defaults + grid) <- CLI
        full_data_params  = merge_defaults(tmp_data_params, parsed)
        full_model_params = merge_defaults(tmp_model_params, parsed)
        full_algo_params  = merge_defaults(tmp_algo_params, parsed)

        print(full_data_params,
              full_model_params,
              full_algo_params)

        # create a new instance of a model
        def model_fn():
            return ModelCls(**full_model_params)
        
        # flatten dicts
        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, key).items())
                else:
                    items.append((key, v))
            return dict(items)

        data = DataCls(**full_data_params)
        algo = AlgoCls(model_fn=model_fn, **full_algo_params) # init fed algo
        final_model = algo.run(data)

        # collect metrics
        mse = algo.MSE.detach().cpu().numpy()          # shape = (R,)
        loss_hist = algo.loss_history.detach().cpu().numpy()  # shape = (n_clients, R)
        
        # save data 
        flat_data  = flatten_dict(full_data_params, parent_key="data")
        flat_model = flatten_dict(full_model_params, parent_key="model")
        flat_algo  = flatten_dict(full_algo_params, parent_key="algo")

        # Add one row per round
        for r in range(len(mse)):
            row = {
                "rep": rep,
                "iter": r,          # iteration number
                "mse": mse[r],
                "loss_mean": loss_hist[:, r].mean()  # optional: avg across clients
            }
            row.update(flat_data)
            row.update(flat_model)
            row.update(flat_algo)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("experiment_results.csv", index=False)
    print(f"Saved results to experiment_results.csv")

