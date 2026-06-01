import torch
import numpy as np
import random
from torch import nn
from utils.metrics import MSE, MSE_params, accuracy, F1


class Algorithm1_UCB:
    """
    Algorithm1 with UCB-guided candidate selection.

    Instead of uniformly sampling S candidates each round, maintains a per-client
    reward history and selects the top-S candidates by UCB score:

        score(i, j, t) = mean_reward(i, j) + c * sqrt(ln(t) / n_tried(i, j))

    Untried candidates always get priority (optimistic initialisation, score = inf).
    After probing the S candidates, rewards are observed and the estimates updated.

    Also includes:
    - Optional proximal regularisation (mu > 0) to prevent local training loss from
      collapsing to zero (keeping the reward signal alive).
    - BN-stats restoration before candidate eval (correctness fix vs Algorithm1).
    """

    def __init__(
            self,
            model_fn,
            loss_fn,
            metrics={"MSE_test": MSE},
            R=50,
            R_local=5,
            S=20,
            lrate=0.01,
            lrate_decay=None,
            mu=0.0,
            ucb_c=1.0,
            device='cpu',
            seed=None,
            dynamic=False,
            n_clusters=None,
            n_active_clusters=None,
            cluster_rotation_freq=1,
    ):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.R = R
        self.R_local = R_local
        self.S = S
        self.lrate_init = lrate
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.mu = mu
        self.ucb_c = ucb_c
        self.device = device
        self.seed = seed

        self.dynamic = dynamic
        self.n_clusters = n_clusters
        self.n_active_clusters = n_active_clusters
        self.cluster_rotation_freq = cluster_rotation_freq

        self.client_models = None
        self.candidate_model = self.model_fn().to(self.device)
        self.loss_history = None
        self.metrics_history = {name: torch.zeros(self.R, device=self.device) for name in metrics.keys()}

        # UCB state — initialised in run() once n_clients is known
        self._reward_mean = None   # shape (n_clients, n_clients)
        self._n_tried     = None   # shape (n_clients, n_clients)

    # ------------------------------------------------------------------ #
    # Seed / inference helpers
    # ------------------------------------------------------------------ #

    def set_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.device != 'cpu':
                torch.cuda.manual_seed_all(self.seed)

    def get_predictions(self, model, X):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model(X)
        if was_training:
            model.train()
        return out

    def sample_clients(self, n_clients, cluster_labels, active_clusters, device):
        selected = torch.tensor(
            [i for i in range(n_clients) if cluster_labels[i].item() in active_clusters],
            device=device,
        )
        return selected

    # ------------------------------------------------------------------ #
    # UCB helpers
    # ------------------------------------------------------------------ #

    def _ucb_select(self, client_idx, pool, round_idx):
        """
        Select min(S, |pool|) candidates from pool using UCB scores.

        Untried candidates (n_tried == 0) get score = inf so they are always
        preferred; among them the ordering is randomised.  Once every candidate
        has been tried at least once, selection is driven purely by UCB.

        Args:
            client_idx : int
            pool       : list[int]  — eligible candidate indices
            round_idx  : int        — current round (0-based), used for ln(t)

        Returns:
            list[int]  — selected candidate indices (length min(S, |pool|))
        """
        n_pool = len(pool)
        k = min(self.S, n_pool)
        if k == n_pool:
            return list(pool)

        log_t = np.log(round_idx + 1)   # ln(t), t >= 1
        scores = np.empty(n_pool)

        for idx, j in enumerate(pool):
            n = self._n_tried[client_idx, j]
            if n == 0:
                # Large base + uniform noise so untried candidates are shuffled
                scores[idx] = 1e9 + self._rng.random()
            else:
                scores[idx] = (self._reward_mean[client_idx, j]
                               + self.ucb_c * np.sqrt(log_t / n))

        top_k = np.argpartition(-scores, k - 1)[:k]
        return [pool[i] for i in top_k]

    def _update_ucb(self, client_idx, candidate_indices, rewards):
        """
        Incremental mean update for all probed candidates.

        Args:
            client_idx       : int
            candidate_indices: list[int]
            rewards          : list[float]  — r_j = baseline_loss - loss_after_j
        """
        for j, r in zip(candidate_indices, rewards):
            n = self._n_tried[client_idx, j]
            self._reward_mean[client_idx, j] = (
                (self._reward_mean[client_idx, j] * n + r) / (n + 1)
            )
            self._n_tried[client_idx, j] += 1

    # ------------------------------------------------------------------ #
    # Training helpers
    # ------------------------------------------------------------------ #

    def _proximal_term(self, model, anchor_params):
        prox = torch.tensor(0.0, device=self.device)
        for name, p in model.named_parameters():
            if name in anchor_params:
                prox = prox + torch.sum((p - anchor_params[name]) ** 2)
        return prox

    def local_train(self, model, X, y, anchor_params=None):
        model.train()
        data_size = X.shape[0]
        batch_size = min(32, data_size)

        X_i = X.to(self.device)
        y_i = y.to(self.device)

        for _ in range(self.R_local):
            perm = torch.randperm(data_size, device=X_i.device)
            X_s = X_i[perm]
            y_s = y_i[perm]

            for start in range(0, data_size, batch_size):
                end = start + batch_size
                X_b = X_s[start:end]
                y_b = y_s[start:end]

                pred = model(X_b)
                loss = self.loss_fn(pred, y_b)

                if self.mu > 0.0 and anchor_params is not None:
                    loss = loss + (self.mu / 2.0) * self._proximal_term(model, anchor_params)

                grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
                with torch.no_grad():
                    for p, g in zip(model.parameters(), grads):
                        p -= self.lrate * g

    def load_bn_state(self, model, bn_state):
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.copy_(bn_state[name][0])
                m.running_var.copy_(bn_state[name][1])

    # ------------------------------------------------------------------ #
    # Weight update — returns per-candidate losses for UCB update
    # ------------------------------------------------------------------ #

    def weight_update(self, model, X_candidates, y_candidates, X_train_i, y_train_i):
        """
        Train on each of the S candidate datasets starting from the client's
        current model, evaluate loss on the client's own training data, and
        return the best params together with per-candidate losses.

        Returns:
            best_params        : state_dict (cpu)
            best_loss          : scalar tensor
            per_cand_losses    : list[float]  length == len(X_candidates)
        """
        device = self.device
        X_train_i = X_train_i.to(device)
        y_train_i = y_train_i.to(device)

        S = len(X_candidates)

        base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        client_bn_stats = {
            name: (mod.running_mean.clone(), mod.running_var.clone())
            for name, mod in model.named_modules()
            if isinstance(mod, nn.BatchNorm2d)
        }

        cand_anchor = None
        if self.mu > 0.0:
            cand_anchor = {
                name: base_state[name].to(device)
                for name, _ in model.named_parameters()
            }

        candidate_model = self.candidate_model

        best_loss  = torch.tensor(float("inf"), device=device)
        best_params = None
        per_cand_losses = []

        for k in range(S):
            candidate_model.load_state_dict(base_state, strict=True)
            self.load_bn_state(candidate_model, client_bn_stats)
            candidate_model.train()

            data_size  = X_candidates[k].shape[0]
            batch_size = min(32, data_size)
            X_k = X_candidates[k].to(device)
            y_k = y_candidates[k].to(device)

            for _ in range(self.R_local):
                perm = torch.randperm(data_size, device=X_k.device)
                X_s = X_k[perm]
                y_s = y_k[perm]

                for start in range(0, data_size, batch_size):
                    end = start + batch_size
                    X_b = X_s[start:end]
                    y_b = y_s[start:end]

                    pred = candidate_model(X_b)
                    loss = self.loss_fn(pred, y_b)

                    if self.mu > 0.0 and cand_anchor is not None:
                        loss = loss + (self.mu / 2.0) * self._proximal_term(candidate_model, cand_anchor)

                    grads = torch.autograd.grad(loss, candidate_model.parameters(), create_graph=False)
                    with torch.no_grad():
                        for p, g in zip(candidate_model.parameters(), grads):
                            p -= self.lrate * g

                del X_b, y_b, pred, loss, grads

            # Restore client BN stats before eval for a fair loss estimate
            self.load_bn_state(candidate_model, client_bn_stats)
            candidate_model.eval()
            with torch.no_grad():
                pred = candidate_model(X_train_i)
                loss = self.loss_fn(pred, y_train_i)

            per_cand_losses.append(loss.item())

            if loss < best_loss:
                best_loss = loss
                best_params = {k_: v.clone().cpu() for k_, v in candidate_model.state_dict().items()}

        return best_params, best_loss, per_cand_losses

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self, data):
        self.set_seed()
        self._rng = np.random.default_rng(self.seed)

        X_train, y_train = data["train"]
        X_test,  y_test  = data["test"]
        cluster_labels = data.get("cluster_labels", None)
        true_weights   = data.get("true_weights",   None)

        device    = self.device
        n_clients = X_train.shape[0]

        X_train = X_train.cpu()
        y_train = y_train.cpu()
        X_test  = X_test.cpu()
        y_test  = y_test.cpu()

        # Initialise client models
        self.client_models = [self.model_fn().to(device) for _ in range(n_clients)]
        self.loss_history  = np.zeros((n_clients, self.R))

        # Initialise UCB tables
        self._reward_mean = np.zeros((n_clients, n_clients), dtype=np.float64)
        self._n_tried     = np.zeros((n_clients, n_clients), dtype=np.int64)

        if cluster_labels is not None:
            cluster_labels = torch.as_tensor(cluster_labels, device=device)
        if true_weights is not None:
            true_weights = torch.as_tensor(true_weights, device=device)

        if self.dynamic and cluster_labels is not None:
            n_clust  = self.n_clusters or (int(cluster_labels.max().item()) + 1)
            n_active = min(self.n_active_clusters or (n_clust - 1), n_clust)
            use_dynamic = True
        else:
            use_dynamic = False

        active_clusters = None

        for r in range(self.R):
            if self.lrate_decay is not None:
                self.lrate = self.lrate_init * (self.lrate_decay ** r)

            # --- Determine active clients ---
            if use_dynamic:
                if active_clusters is None or r % self.cluster_rotation_freq == 0:
                    active_clusters = set(
                        torch.randperm(n_clust, device=device)[:n_active].tolist()
                    )
                selected_clients = self.sample_clients(n_clients, cluster_labels, active_clusters, device)
            else:
                selected_clients = torch.arange(n_clients, device=device)

            selected_list = selected_clients.tolist()

            # --- Local updates ---
            for i in selected_list:
                anchor = None
                if self.mu > 0.0:
                    anchor = {name: p.detach().clone()
                              for name, p in self.client_models[i].named_parameters()}
                self.local_train(self.client_models[i], X_train[i], y_train[i], anchor)

            # --- UCB candidate selection + weight update ---
            for i in selected_list:
                pool = [j for j in selected_list if j != i]
                if not pool:
                    continue

                # UCB: pick which S candidates to probe
                chosen = self._ucb_select(i, pool, r)

                # Compute baseline loss on client i's data before any candidate update
                self.client_models[i].eval()
                with torch.no_grad():
                    baseline_loss = self.loss_fn(
                        self.client_models[i](X_train[i].to(device)),
                        y_train[i].to(device),
                    ).item()

                # Probe the chosen candidates
                best_params, best_loss, per_cand_losses = self.weight_update(
                    self.client_models[i],
                    [X_train[j] for j in chosen],
                    [y_train[j] for j in chosen],
                    X_train[i],
                    y_train[i],
                )

                # UCB update: reward_j = reduction in loss from using candidate j
                rewards = [baseline_loss - lj for lj in per_cand_losses]
                self._update_ucb(i, chosen, rewards)

                self.loss_history[i, r] = best_loss.detach().cpu().item()
                self.client_models[i].load_state_dict(
                    {k: v.to(device) for k, v in best_params.items()}
                )

            # --- Evaluate all clients ---
            metrics_sums = {name: torch.tensor(0.0, device=device) for name in self.metrics.keys()}
            for i in range(n_clients):
                test_predictions = None
                for metric_name, metric_fn in self.metrics.items():
                    if metric_name == "MSE_params":
                        if true_weights is None or cluster_labels is None:
                            raise ValueError("MSE_params requires true_weights and cluster_labels")
                        cluster_id  = cluster_labels[i]
                        param_tensor = list(self.client_models[i].state_dict().values())[0]
                        metric_value = metric_fn(torch.squeeze(param_tensor), true_weights[cluster_id])
                    else:
                        if test_predictions is None:
                            X_test_i = X_test[i].to(device)
                            y_test_i = y_test[i].to(device)
                            test_predictions = self.get_predictions(self.client_models[i], X_test_i)
                        metric_value = metric_fn(test_predictions, y_test_i)
                    metrics_sums[metric_name] += metric_value.detach()

                try:
                    del X_test_i, y_test_i, test_predictions
                except NameError:
                    pass

            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = metrics_sums[metric_name] / n_clients

        return self.client_models
