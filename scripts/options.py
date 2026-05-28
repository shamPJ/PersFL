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
    parser.add_argument("--no_scale", action="store_true", default=False, help="disable StandardScaler on features")
    parser.add_argument("--sigma", type=float, default=0.0, help="within-cluster rotation spread (std dev in degrees)")
    parser.add_argument("--shift_at", type=int, default=None, help="round at which distribution shift occurs (cifar10_shifted)")
    parser.add_argument("--R", type=int, default=1500)
    parser.add_argument("--R_local", type=int, default=0)
    parser.add_argument("--lrate", type=float, default=0.01)
    parser.add_argument("--lrate_decay", type=float, default=None)
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--lmbd", type=float, default=1)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--ucb_c", type=float, default=1.0)
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--algo_n_clusters", type=int, default=None)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_samples_test", type=int, default=500)
    parser.add_argument("--S", type=int, default=20, help="canidate set size")
    parser.add_argument("--K", type=int, default=4, help="top-K candidates to aggregate (Algorithm1_TopK)")
    parser.add_argument("--weighting", type=str, default="uniform", choices=["uniform", "reward"], help="aggregation weighting scheme")
    parser.add_argument("--temperature", type=float, default=1.0, help="inverse temperature for reward-weighted aggregation")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--problem", type=str, choices=["regression", "classification"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fname", type=str, default="results.csv")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    return vars(args)
