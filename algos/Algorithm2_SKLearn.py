import numpy as np
import torch
import random
from utils.metrics import MSE, accuracy, F1
from joblib import Parallel, delayed
import os

class Algorithm2_SKLearn:
    def __init__(self, model_fn, ..., n_jobs=None):
        self.n_jobs = n_jobs if n_jobs is not None else self._get_default_n_jobs()

    
    
class Algorithm2_SKLearn:
    """
    Algorithm2 adapted for sklearn-style models (DecisionTree, etc.)
    """

    def __init__(self, model_fn, loss_fn=None, metrics={"MSE_val": MSE}, R=50, S=20, lmbd=0.05, seed=None, device=None, n_jobs=None):
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
        self.n_jobs = n_jobs if n_jobs is not None else self._get_default_n_jobs()

        self.client_models = None
        self.loss_history = None  # shape (n_clients, R)
        self.metrics_history = {name: np.zeros(R) for name in metrics.keys()}

    def set_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
    
    def _get_default_n_jobs(self):
        return int(
            os.environ.get(
                "N_JOBS",
                os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
            )
        )
    
    def fit_candidate(self, j, X_cand, y_cand, X_test, y_test):
        """
        Fit a candidate model for client i using augmented data
        """
        new_model = self.model_fn(seed=self.seed+j)
        X_aug = np.concatenate([X_cand, X_test])
        y_aug = np.concatenate([
            y_cand.reshape(-1),
            y_test.reshape(-1)
        ])

        m, m_test = X_cand.shape[0], X_test.shape[0]
        sample_weight = np.concatenate((np.ones((m,)), np.ones((m_test,)) * self.lmbd))

        new_model.fit(X_aug, y_aug, sample_weight=sample_weight)
        return new_model

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

        X_test = np.random.normal(0, 1, size=(100, X_train.shape[2]))  # fixed test for data augmentation

        n_clients = X_train.shape[0]
        self.client_models = [self.model_fn(seed=self.seed+i) for i in range(n_clients)]
        self.loss_history = np.zeros((n_clients, self.R))

        # init models with local training
        for i in range(n_clients):
                self.client_models[i].fit(X_train[i], y_train[i])  # local training step

        for r in range(self.R):
            # Step 2: sample candidate neighbors
            candidate_indices = []
            for i in range(n_clients):
                pool = np.concatenate([np.arange(i), np.arange(i+1, n_clients)])
                idx = np.random.choice(pool, min(self.S, len(pool)), replace=False)
                candidate_indices.append(idx)

            # Step 3: generate candidate models
            all_candidate_models = []
            for i in range(n_clients):
                y_test = self.client_models[i].predict(X_test)  # pseudo-labels for test set
                X_cand = X_train[candidate_indices[i]]
                y_cand = y_train[candidate_indices[i]]
                candidates = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.fit_candidate)(j, X_cand[j], y_cand[j], X_test, y_test)
                        for j in range(len(X_cand))               
                )
                all_candidate_models.append(candidates)

            # Step 4: select best candidates
            for i in range(n_clients):
                losses = []
                for candidate in all_candidate_models[i]:
                    pred = candidate.predict(X_train[i])
                    # MSE loss works on torch tensors
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