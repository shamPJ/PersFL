from algos.flavg import FedAvg
import torch
from dataclasses import dataclass
from torch import nn
from typing import Dict, Any
from utils.metrics import MSE, MSE_params, accuracy, F1

@dataclass
class ModelSpec:
    module: str
    cls: str
    default_params: Dict[str, Any]

@dataclass
class DatasetSpec:
    module: str
    loader: str
    default_params: Dict[str, Any]

@dataclass
class AlgoSpec:
    module: str
    cls: str
    default_params: Dict[str, Any]

# Registries
MODELS = {
    "linreg": ModelSpec(module="model.linreg", cls="LinReg", default_params={"n_features": 2, "bias": False}),
    "cnn": ModelSpec(module="model.cnn", cls="CNN", default_params={"input_shape": (3, 32, 32), "num_classes": 10})
}

DATASETS = {
    "synthetic": DatasetSpec(module="data.synthetic", loader="generate_data", default_params={
        "n_clusters": 2,
        "n_clients": 100,
        "n_samples": 10,
        "n_samples_val": 500,
        "n_features": 2,
        "noise_scale": 0,
        "noise_weight": 0 
    })
}

ALGOS = {
    "persfl": AlgoSpec(module="algos.persfl", cls="PersFL", default_params={
        "lrate": 0.03,
        "S": 30,
        "R": 1500,
        "R_local": 0}),
    
    "fedavg": AlgoSpec(module="algos.fedavg", cls="FedAvg", default_params={
        "lrate": 0.03,
        "S": 30,
        "R": 1500})
}
