from dataclasses import dataclass
from torch import nn
from typing import Dict, Any

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
        "n_features": 2,
        "noise_scale": 0
    })
}

ALGOS = {
    "persfl": AlgoSpec(module="algos.persfl", cls="PersFL", default_params={
        "loss_fn": nn.MSELoss(), 
        "lrate": 0.03,
        "S": 20,
        "R": 1500})
}
