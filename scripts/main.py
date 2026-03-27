from registry import MODELS, DATASETS, ALGOS
from utils.metrics import MSE, MSE_params, accuracy, F1
from options import read_options
from itertools import product
from torch import nn
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

if __name__ == "__main__":
    
    parsed = read_options()
    fname = parsed["fname"]

    device = torch.device(parsed["device"])
    R = parsed["R"]
    seed = parsed["seed"]

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

    # -----------------------------
    # Detect problem type
    # -----------------------------
    if parsed.get("problem"):
        problem_type = parsed["problem"]
    else:
        problem_type = getattr(data_spec, "problem_type", "regression")  # fallback

    print(f"Detected problem type: {problem_type}")

    # -----------------------------
    # Choose metrics automatically
    # -----------------------------
    metrics = {}
    if problem_type == "regression":
        loss_fn = nn.MSELoss()
        metrics = {"MSE_val": MSE, "MSE_params": MSE_params}

    elif problem_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
        metrics = {"accuracy": accuracy}

    # -----------------------------
    # Model factory
    # -----------------------------
    def model_fn():
        return ModelCls(**full_model_params)

    # -----------------------------
    # Load dataset
    # -----------------------------
    data = DataCls(**full_data_params)

    # -----------------------------
    # Initialize federated algorithm
    # -----------------------------
    algo = AlgoCls(
        model_fn=model_fn,
        loss_fn=loss_fn, 
        metrics=metrics, 
        device=device,
        seed=seed,
        **full_algo_params
    )

    # -----------------------------
    # Run federated learning
    # -----------------------------
    final_models = algo.run(data)

    # -----------------------------
    # Collect metrics
    # -----------------------------
    loss_hist = algo.loss_history.detach().cpu().numpy()  # shape = (n_clients, R)
    
    # save data 
    flat_data  = flatten_dict(full_data_params, parent_key="data")
    flat_model = flatten_dict(full_model_params, parent_key="model")
    flat_algo  = flatten_dict(full_algo_params, parent_key="algo")

    rows = []
    for r in range(R):
        row = {
            "iter": r,
            "loss_mean": loss_hist[:, r].mean()
        }

        # Add metrics from dictionary of tensors algo.metrics_history if it exists
        if hasattr(algo, "metrics_history") and algo.metrics_history is not None:
            for metric_name, metric_tensor in algo.metrics_history.items():
                row[metric_name] = metric_tensor[r].item()  # convert each value to Python scalar

        row.update(flat_data)
        row.update(flat_model)
        row.update(flat_algo)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(fname, index=False)

    print(f"Saved results to {fname}")
