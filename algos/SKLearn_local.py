import numpy as np
import torch
import random
from utils.metrics import MSE, accuracy, F1

class SKLearn_local:
    """
    Sklearn-style local training (no communication, no candidate neighbors)
    """

    def __init__(self, model_fn, loss_fn=None, metrics={"MSE_val": MSE}, seed=None, device=None):
        """
        Args:
            model_fn: callable returning fresh sklearn model instance
            loss_fn: callable for loss (optional, e.g. MSE)
            metrics: dict of metric functions
            R: number of iterations
            S: number of candidate neighbors per client
            seed: random seed
        """
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.seed = seed

        self.client_models = None
        self.loss_history = None  # shape (n_clients, R)
        self.metrics_history = {name: np.zeros(R) for name in metrics.keys()}

    def set_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def run(self, data):
        """
        Run Algorithm2 for sklearn models
        Args:
            data: dict with "train": (X_train, y_train), "val": (X_val, y_val)
                  X_train: shape (n_clients, m_i, d), y_train: shape (n_clients, m_i)
        Returns:
            client_models: list of trained models
        """
        self.set_seed()
        X_train, y_train = data["train"] # shapes: (n_clients, m_i, d), (n_clients, m_i)
        X_val, y_val = data["val"]

        n_clients = X_train.shape[0]
        self.client_models = [self.model_fn(seed=self.seed+i) for i in range(n_clients)]
        self.loss_history = np.zeros((n_clients, self.R))

        for r in range(self.R):
            # Step 1: local updates
            for i in range(n_clients):
                self.client_models[i].fit(X_train[i], y_train[i])  
                pred_train = self.client_models[i].predict(X_train[i])
                loss = self.loss_fn(torch.from_numpy(pred_train).reshape(-1,1), y_train[i].reshape(-1,1)).item()
                self.loss_history[i, r] = loss

            # Step 2: evaluate metrics on validation set
            metrics_sums = {name: 0.0 for name in self.metrics.keys()}
            for i in range(n_clients):
                val_pred = self.client_models[i].predict(X_val[i])
                for metric_name, metric_fn in self.metrics.items():
                    metrics_sums[metric_name] += metric_fn(torch.from_numpy(val_pred).reshape(-1,1), y_val[i].reshape(-1,1)).item()

            # store average metrics
            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = metrics_sums[metric_name] / n_clients

        return self.client_models