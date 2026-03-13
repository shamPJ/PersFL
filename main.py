from registry import MODELS, DATASETS, ALGOS
from options import read_options
from itertools import product
import pandas as pd
import importlib
import torch
import random
import numpy as np
import os

def load_object(module, name):
    # e.g. for model/linreg.py with udf LinReg() - module = 'model.linreg' and 'name'='LinReg'
    mod = importlib.import_module(module)
    return getattr(mod, name)

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
    fname = parsed["fname"]

    device = torch.device(parsed["device"])

    model_spec = MODELS[parsed["model"]] # choose dataset, model, algo from user input or default vals
    data_spec = DATASETS[parsed["dataset"]] # e.g. 'linreg', 'synthetic', 'persfl'; corresponds to .py files
    algo_spec = ALGOS[parsed["algo"]]
    print(model_spec.module, model_spec.cls)
    ModelCls = load_object(model_spec.module, model_spec.cls)
    AlgoCls = load_object(algo_spec.module, algo_spec.cls)
    DataCls = load_object(data_spec.module, data_spec.loader) # data gen function or pytorch data loader

    # --------------------------------
    # Merge CLI parameters with defaults
    # --------------------------------
    full_data_params  = merge_defaults(data_spec.default_params, parsed)
    full_model_params = merge_defaults(model_spec.default_params, parsed)
    full_algo_params  = merge_defaults(algo_spec.default_params, parsed)

    print("Data params:", full_data_params)
    print("Model params:", full_model_params)
    print("Algo params:", full_algo_params)

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
    algo = AlgoCls(model_fn=model_fn, device=device, **full_algo_params) # init fed algo
    final_model = algo.run(data)

    # collect metrics
    mse = algo.MSE.detach().cpu().numpy()          # shape = (R,)
    loss_hist = algo.loss_history.detach().cpu().numpy()  # shape = (n_clients, R)
    
    # save data 
    flat_data  = flatten_dict(full_data_params, parent_key="data")
    flat_model = flatten_dict(full_model_params, parent_key="model")
    flat_algo  = flatten_dict(full_algo_params, parent_key="algo")

    rows = []
    for r in range(len(mse)):
        row = {
            "iter": r,
            "mse": mse[r],
            "loss_mean": loss_hist[:, r].mean()
        }

        row.update(flat_data)
        row.update(flat_model)
        row.update(flat_algo)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(fname, index=False)

    print(f"Saved results to {fname}")

