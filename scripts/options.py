# options.py
import argparse
from registry import MODELS, DATASETS, ALGOS

def read_options():
    parser = argparse.ArgumentParser()

    # dynamically populate choices from registries
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        default="linreg",  
        help="model to use"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        default="synthetic",
        help="dataset to use"
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=list(ALGOS.keys()),
        default="Algorithm1",
        help="federated algorithm to use"
    )

    # generic options
    parser.add_argument("--noise_scale", type=float, default=0)
    parser.add_argument("--noise_weight", type=float, default=0)
    parser.add_argument("--R", type=int, default=1500)
    parser.add_argument("--R_local", type=int, default=0)
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--lrate", type=float, default=0.1)
    parser.add_argument("--lmbd", type=float, default=1)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--lrate_decay", type=float, default=None)
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_samples_val", type=int, default=500)
    parser.add_argument("--S", type=int, default=20, help="canidate set size")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--problem", type=str, choices=["regression", "classification"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fname", type=str, default="results.csv")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    return vars(args)
