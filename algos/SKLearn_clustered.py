import numpy as np
import torch
import random
from utils.metrics import MSE, accuracy, F1

class Algorithm2_SKLearn:
    """
    Algorithm2 adapted for sklearn-style models (DecisionTree, etc.)
    """

    def __init__(self, model_fn, loss_fn=None, metrics={"MSE_val": MSE}, R=50, S=20, lmbd=1, seed=None, device=None):
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
        self.R = R
        self.S = S
        self.lmbd = lmbd
        self.seed = seed

        self.client_models = None
        self.loss_history = None  # shape (n_clients, R)
        self.metrics_history = {name: np.zeros(R) for name in metrics.keys()}

    def set_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def hypothesis_update(self, model, X_candidates, y_candidates, X_test):
        """
        Generate candidate models by re-fitting on candidate datasets
        """
        candidate_models = []
        for i in range(len(X_candidates)):
            new_model = self.model_fn(seed=self.seed+i)
            X_aug = np.concatenate([X_candidates[i], X_test])
            y_aug = np.concatenate([y_candidates[i].reshape(-1), model.predict(X_test).reshape(-1)])
            new_model.fit(X_aug, y_aug)
            candidate_models.append(new_model)
        return candidate_models

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

        X_test = np.random.rand(100, X_train.shape[2])  # fixed test for data augmentation

        n_clients = X_train.shape[0]
        self.client_models = [self.model_fn(seed=self.seed+i) for i in range(n_clients)]
        self.loss_history = np.zeros((n_clients, self.R))

        for r in range(self.R):
            # Step 1: local updates
            for i in range(n_clients):
                self.client_models[i].fit(X_train[i], y_train[i])  # local training step

            # Step 2: sample candidate neighbors
            candidate_indices = []
            for i in range(n_clients):
                pool = np.concatenate([np.arange(i), np.arange(i+1, n_clients)])
                idx = np.random.choice(pool, min(self.S, len(pool)), replace=False)
                candidate_indices.append(idx)

            # Step 3: generate candidate models
            all_candidate_models = []
            for i in range(n_clients):
                X_cand = X_train[candidate_indices[i]]
                y_cand = y_train[candidate_indices[i]]
                candidates = self.hypothesis_update(self.client_models[i], X_cand, y_cand, X_test)
                all_candidate_models.append(candidates) # list of lists of models

            # Step 4: select best candidates
            for i in range(n_clients):
                losses = []
                for candidate in all_candidate_models[i]:
                    pred = candidate.predict(X_train[i])
                    # loss works on torch tensors
                    loss = self.loss_fn(torch.from_numpy(pred).reshape(-1,1), y_train[i].reshape(-1,1)).item()
                    losses.append(loss)

                best_idx = np.argmin(losses)
                self.client_models[i] = all_candidate_models[i][best_idx]
                self.loss_history[i, r] = losses[best_idx]

            # Step 5: evaluate metrics on validation set
            metrics_sums = {name: 0.0 for name in self.metrics.keys()}
            for i in range(n_clients):
                val_pred = self.client_models[i].predict(X_val[i])
                for metric_name, metric_fn in self.metrics.items():
                    metrics_sums[metric_name] += metric_fn(torch.from_numpy(val_pred).reshape(-1,1), y_val[i].reshape(-1,1)).item()

            # store average metrics
            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = metrics_sums[metric_name] / n_clients

        return self.client_models