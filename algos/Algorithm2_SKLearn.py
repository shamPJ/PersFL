import numpy as np
import random
from utils.metrics import MSE, accuracy, F1

class Algorithm2_SKLearn:
    """
    Algorithm2 adapted for sklearn-style models (DecisionTree, etc.)
    """

    def __init__(self, model_fn, loss_fn=None, metrics={"MSE_val": MSE}, R=50, S=20, seed=None):
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
        self.seed = seed

        self.client_models = None
        self.loss_history = None  # shape (n_clients, R)
        self.metrics_history = {name: np.zeros(R) for name in metrics.keys()}

    def set_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def local_train(self, model, X, y):
        """Fit model on local client data"""
        model.fit(X, y)

    def get_predictions(self, model, X):
        return model(X)

    def hypothesis_update(self, model, X_candidates, y_candidates):
        """
        Generate candidate models by re-fitting on candidate datasets
        """
        candidate_models = []
        for i in range(len(X_candidates)):
            new_model = self.model_fn()
            new_model.fit(X_candidates[i], y_candidates[i])
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
        X_train, y_train = data["train"]
        X_val, y_val = data["val"]

        n_clients = X_train.shape[0]
        self.client_models = [self.model_fn() for _ in range(n_clients)]
        self.loss_history = np.zeros((n_clients, self.R))

        for r in range(self.R):
            # Step 1: local updates
            for i in range(n_clients):
                self.local_train(self.client_models[i], X_train[i], y_train[i])

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
                candidates = self.hypothesis_update(self.client_models[i], X_cand, y_cand)
                all_candidate_models.append(candidates)

            # Step 4: evaluate candidates
            metrics_sums = {name: 0.0 for name in self.metrics.keys()}
            for i in range(n_clients):
                losses = []
                for candidate in all_candidate_models[i]:
                    pred = self.get_predictions(candidate, X_train[i])
                    if self.loss_fn:
                        loss = self.loss_fn(pred, y_train[i])
                    else:
                        loss = np.mean((pred - y_train[i])**2)
                    losses.append(loss)

                best_idx = np.argmin(losses)
                self.client_models[i] = all_candidate_models[i][best_idx]
                self.loss_history[i, r] = losses[best_idx]

                # evaluate metrics on validation set
                val_pred = self.get_predictions(self.client_models[i], X_val[i])
                for metric_name, metric_fn in self.metrics.items():
                    metrics_sums[metric_name] += metric_fn(val_pred, y_val[i])

            # store average metrics
            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = metrics_sums[metric_name] / n_clients

        return self.client_models