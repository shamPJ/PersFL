import numpy as np
import torch
import random
from utils.metrics import MSE, accuracy, F1
from joblib import Parallel, delayed
import os

class Algorithm2_SKLearn:
    """
    Algorithm2 adapted for sklearn-style models (DecisionTree, etc.)
    Supports dynamic cluster participation: each iteration sample only clients from
    a subset of clusters, mirroring Algorithm1's dynamic mode.
    """

    def __init__(
        self,
        model_fn,
        loss_fn=None,
        metrics={"MSE_test": MSE},
        R=50,
        S=20,
        lmbd=0.05,
        seed=None,
        device=None,
        n_jobs=None,
        # --- dynamic cluster participation ---
        dynamic=False,
        n_clusters=None,
        n_active_clusters=None,
        cluster_rotation_freq=1,
    ):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.R = R
        self.S = S
        self.lmbd = lmbd
        self.seed = seed
        self.n_jobs = n_jobs if n_jobs is not None else self._get_default_n_jobs()

        self.dynamic = dynamic
        self.n_clusters = n_clusters
        self.n_active_clusters = n_active_clusters
        self.cluster_rotation_freq = cluster_rotation_freq

        self.client_models = None
        self.loss_history = None
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

    def sample_clients(self, n_clients, cluster_labels, active_clusters):
        """Return indices of clients whose cluster is in active_clusters."""
        return [i for i in range(n_clients) if cluster_labels[i] in active_clusters]

    def fit_candidate(self, j, X_cand, y_cand, X_pub, y_pub):
        """Fit a candidate model using augmented data (candidate data + pseudo-labelled public data)."""
        seed_j = (self.seed + j) if self.seed is not None else None
        new_model = self.model_fn(seed=seed_j)

        X_aug = np.concatenate([X_cand, X_pub])
        y_aug = np.concatenate([y_cand.reshape(-1), y_pub.reshape(-1)])

        m, m_pub = X_cand.shape[0], X_pub.shape[0]
        sample_weight = np.concatenate([np.ones(m)*self.lmbd, np.ones(m_pub)])

        new_model.fit(X_aug, y_aug, sample_weight=sample_weight)
        return new_model

    def _eval_loss(self, model, X, y):
        pred = model.predict(X)
        return self.loss_fn(
            torch.from_numpy(pred).reshape(-1, 1),
            torch.as_tensor(y).reshape(-1, 1),
        ).item()

    def _eval_metric(self, model, X, y):
        pred = model.predict(X)
        results = {}
        for name, fn in self.metrics.items():
            results[name] = fn(
                torch.from_numpy(pred).reshape(-1, 1),
                torch.as_tensor(y).reshape(-1, 1),
            ).item()
        return results

    def run(self, data):
        self.set_seed()

        X_train, y_train = data["train"]
        X_test, y_test = data["test"]
        cluster_labels = data.get("cluster_labels", None)

        # Convert cluster_labels to a plain numpy int array for easy indexing
        if cluster_labels is not None:
            if isinstance(cluster_labels, torch.Tensor):
                cluster_labels = cluster_labels.cpu().numpy()
            cluster_labels = np.asarray(cluster_labels, dtype=int)

        n_clients = X_train.shape[0]
        # public dataset is global and same for all iterations
        X_pub = np.random.normal(0, 1, size=(100, X_train.shape[2]))

        self.client_models = [
            self.model_fn(seed=(self.seed + i) if self.seed is not None else None)
            for i in range(n_clients)
        ]
        self.loss_history = np.zeros((n_clients, self.R))

        # Initial local fit for every client
        for i in range(n_clients):
            self.client_models[i].fit(X_train[i], y_train[i])

        # --- resolve dynamic cluster config ---
        if self.dynamic and cluster_labels is not None:
            n_clust = self.n_clusters or (int(cluster_labels.max()) + 1)
            n_active = min(self.n_active_clusters or (n_clust - 1), n_clust)
            use_dynamic = True
        else:
            use_dynamic = False

        active_clusters = None

        for r in range(self.R):

            # --- determine selected clients for this round ---
            if use_dynamic:
                if active_clusters is None or r % self.cluster_rotation_freq == 0:
                    active_clusters = set(
                        np.random.choice(n_clust, n_active, replace=False).tolist()
                    )
                selected_list = self.sample_clients(n_clients, cluster_labels, active_clusters)
            else:
                selected_list = list(range(n_clients))

            # Step 1: sample candidate neighbours from the selected pool only
            candidate_pairs = []   # list of (client_i, array_of_candidate_indices)
            for i in selected_list:
                pool = [j for j in selected_list if j != i]
                if not pool:
                    continue
                idx = np.random.choice(pool, min(self.S, len(pool)), replace=False)
                candidate_pairs.append((i, idx))

            # Step 2: fit candidate models (only for selected clients)
            all_candidate_models = {}
            for i, cand_idx in candidate_pairs:
                y_pub_i = self.client_models[i].predict(X_pub)
                X_cand = X_train[cand_idx]
                y_cand = y_train[cand_idx]

                candidates = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.fit_candidate)(j, X_cand[j], y_cand[j], X_pub, y_pub_i)
                    for j in range(len(cand_idx))
                )
                all_candidate_models[i] = candidates

            # Step 3: select best candidate (only for selected clients)
            for i, _ in candidate_pairs:
                losses = [self._eval_loss(c, X_train[i], y_train[i])
                          for c in all_candidate_models[i]]
                best = int(np.argmin(losses))
                self.client_models[i] = all_candidate_models[i][best]
                self.loss_history[i, r] = losses[best]

            # Step 4: evaluate metrics on ALL clients (including inactive ones)
            metrics_sums = {name: 0.0 for name in self.metrics.keys()}
            for i in range(n_clients):
                for name, val in self._eval_metric(self.client_models[i], X_test[i], y_test[i]).items():
                    metrics_sums[name] += val

            for name in self.metrics.keys():
                self.metrics_history[name][r] = metrics_sums[name] / n_clients
        # Local training, used for baseline
        if self.R == 0:
            self.metrics_history = {name: np.zeros(1) for name in self.metrics.keys()}
            metrics_sums = {name: 0.0 for name in self.metrics.keys()}
            for i in range(n_clients):
                for name, val in self._eval_metric(self.client_models[i], X_test[i], y_test[i]).items():
                    metrics_sums[name] += val
            for name in self.metrics.keys():
                self.metrics_history[name][0] = metrics_sums[name] / n_clients

        return self.client_models
