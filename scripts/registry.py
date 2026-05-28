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
    "linreg": ModelSpec(module="model.linreg", cls="LinReg", default_params={
        "n_features": 2, 
        "bias": False}),
    "cnn": ModelSpec(module="model.cnn", cls="CNN", default_params={
        "input_shape": (3, 32, 32), 
        "n_classes": 10}),
    "decision_tree": ModelSpec(module="model.tree", cls="DecisionTree", default_params={
        "max_depth": 5}),
    "random_forest": ModelSpec(module="model.forest", cls="RandomForest", default_params={
        "max_depth": 10, 
        "n_estimators": 10})
}

DATASETS = {
    "synthetic": DatasetSpec(module="data.synthetic", loader="generate_data", default_params={
        "n_clusters": 2,
        "n_clients": 100,
        "n_samples": 10,
        "n_samples_test": 500,
        "n_features": 2,
        "noise_scale": 0,
        "noise_weight": 0,
        "no_scale": False,
    }),

    "cifar10": DatasetSpec(module="data.cifar10", loader="generate_rotated_cifar10", default_params={
        "n_clients": 200,
        "n_clusters": 4,
        "n_samples": 200,
        "n_samples_test": 1000,
        "sigma": 0.0,
        "seed": 0
    }),
    "cifar10_shifted": DatasetSpec(module="data.cifar10", loader="generate_rotated_cifar10_shifted", default_params={
        "n_clients": 24,
        "n_clusters": 2,
        "n_samples": 500,
        "n_samples_test": 1000,
        "shift_at": 15,
        "seed": 0
    }),
}

ALGOS = {
    "Algorithm1": AlgoSpec(module="algos.Algorithm1", cls="Algorithm1", default_params={
        "lrate": 0.01,
        "lrate_decay": 0.999,
        "S": 30,
        "R": 1500,
        "R_local": 1}),

    "Algorithm2": AlgoSpec(module="algos.Algorithm2", cls="Algorithm2", default_params={
        "lrate": 0.01,
        "lmbd": 1,
        "S": 30,
        "R": 1500,
        "R_local": 1}),

    "Algorithm2_SKLearn": AlgoSpec(module="algos.Algorithm2_SKLearn", cls="Algorithm2_SKLearn", default_params={
        "lmbd": 1,
        "S": 30,
        "R": 100}),
    
    "FedAvg": AlgoSpec(module="algos.FedAvg", cls="FedAvg", default_params={
        "lrate": 0.01,
        "lrate_decay": 0.999,
        "R": 500,
        "R_local": 5
        }),

    "FedBN": AlgoSpec(module="algos.FedBN", cls="FedBN", default_params={
        "lrate": 0.01,
        "lrate_decay": 0.999,
        "R": 500,
        "R_local": 5
        }),

    "FedProx": AlgoSpec(module="algos.FedProx", cls="FedProx", default_params={
        "lrate": 0.01,
        "lrate_decay": 0.999,
        "R": 500,
        "R_local": 10,
        "mu": 0.01
      }),

    "IFCA": AlgoSpec(module="algos.IFCA", cls="IFCA", default_params={
        "lrate": 0.01,
        "lrate_decay": 0.999,
        "n_clusters": 4,
        "algo_n_clusters": None,
        "R": 500,
        "R_local": 10}),

    "Algorithm1_prox": AlgoSpec(module="algos.Algorithm1_prox", cls="Algorithm1_prox", default_params={
        "lrate": 0.01,
        "lrate_decay": 0.999,
        "S": 30,
        "R": 1500,
        "R_local": 1,
        "mu": 0.01}),

    "Algorithm1_TopK": AlgoSpec(module="algos.Algorithm1_TopK", cls="Algorithm1_TopK", default_params={
        "lrate": 0.01,
        "lrate_decay": 0.999,
        "S": 30,
        "K": 4,
        "weighting": "uniform",
        "temperature": 1.0,
        "R": 1500,
        "R_local": 1}),

    "Algorithm1_UCB": AlgoSpec(module="algos.Algorithm1_UCB", cls="Algorithm1_UCB", default_params={
        "lrate": 0.01,
        "lrate_decay": 0.999,
        "S": 30,
        "R": 1500,
        "R_local": 1,
        "mu": 0.0,
        "ucb_c": 1.0}),

    "Ditto": AlgoSpec(module="algos.Ditto", cls="Ditto", default_params={
        "lrate": 0.01,
        "lrate_decay": 0.999,
        "R": 500,
        "R_local": 10,
        "lmbd": 0.1}),
}
