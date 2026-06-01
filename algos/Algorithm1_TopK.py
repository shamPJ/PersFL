import math
import torch
import numpy as np
import random
from torch import nn
from utils.metrics import MSE, MSE_params, accuracy, F1


class Algorithm1_TopK:
    """
    Algorithm 1 with top-K candidate aggregation.

    Instead of selecting the single best candidate (argmin loss), the K
    candidates with the lowest evaluation loss on client i's data are
    averaged together. This recovers FedAvg-like variance reduction when
    the population is homogeneous (most candidates are useful) while
    preserving selectivity when it is heterogeneous (only a few are useful).

    Parameters
    ----------
    K : int
        Number of best candidates to aggregate per round. K=1 recovers the
        original Algorithm 1. K=S recovers uniform averaging over all
        candidates (closest to FedAvg).
    weighting : {'uniform', 'reward'}
        How to weight the top-K candidates when averaging:
        - 'uniform': equal weight 1/K for each candidate.
        - 'reward': softmax over rewards r_{i'} = L_1(ŵ) - L_1(ŵ^{i'})
          scaled by `temperature`. Higher temperature concentrates weight on
          the best candidate; temperature→0 collapses to uniform.
    temperature : float
        Inverse temperature \beta for reward-weighted averaging. Only used when
        weighting='reward'.
    """

    def __init__(
            self,
            model_fn,
            loss_fn,
            metrics={"MSE_test": MSE},
            R=50,
            R_local=5,
            S=20,
            K=4,
            weighting='uniform',
            temperature=1.0,
            lrate=0.01,
            lrate_decay=None,
            device='cpu',
            seed=None
    ):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.R = R
        self.R_local = R_local
        self.S = S
        self.K = K
        self.weighting = weighting
        self.temperature = temperature
        self.lrate_init = lrate
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.device = device
        self.seed = seed

        self.client_models = None
        self.candidate_model = self.model_fn().to(self.device)
        self.loss_history = None
        self.metrics_history = {
            name: torch.zeros(self.R, device=self.device)
            for name in metrics.keys()
        }

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

    def load_bn_state(self, model, bn_state):
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.copy_(bn_state[name][0])
                m.running_var.copy_(bn_state[name][1])

    def local_train(self, model, X, y):
        model.train()

        data_size = X.shape[0]
        batch_size = min(32, data_size)

        X_i = X.to(self.device)
        y_i = y.to(self.device)

        for _ in range(self.R_local):
            perm = torch.randperm(data_size, device=X_i.device)
            X_shuffled = X_i[perm]
            y_shuffled = y_i[perm]

            for start in range(0, data_size, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                pred = model(X_batch)
                loss = self.loss_fn(pred, y_batch)

                grads = torch.autograd.grad(
                    loss, model.parameters(), create_graph=False
                )

                with torch.no_grad():
                    for p, g in zip(model.parameters(), grads):
                        p -= self.lrate * g

    def weight_update(
            self, 
            model, 
            X_candidates, 
            y_candidates,
            X_train_i, 
            y_train_i
        ):

        device = self.device
        X_train_i = X_train_i.to(device)
        y_train_i = y_train_i.to(device)

        S = len(X_candidates)
        K = min(self.K, S)

        base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        client_bn_stats = {
            name: (mod.running_mean.clone(), mod.running_var.clone())
            for name, mod in model.named_modules()
            if isinstance(mod, nn.BatchNorm2d)
        }

        # Baseline loss before any candidate update (used for reward computation)
        model.eval()
        with torch.no_grad():
            baseline_loss = self.loss_fn(model(X_train_i), y_train_i).item()

        candidate_model = self.candidate_model

        candidate_losses = []
        candidate_states = []

        for i in range(S):
            candidate_model.load_state_dict(base_state, strict=True)
            self.load_bn_state(candidate_model, client_bn_stats)
            candidate_model.train()

            data_size = X_candidates[i].shape[0]
            batch_size = min(32, data_size)

            X_i = X_candidates[i].to(device)
            y_i = y_candidates[i].to(device)

            for _ in range(self.R_local):
                perm = torch.randperm(data_size, device=X_i.device)
                X_shuffled = X_i[perm]
                y_shuffled = y_i[perm]

                for start in range(0, data_size, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    pred = candidate_model(X_batch)
                    loss = self.loss_fn(pred, y_batch)

                    grads = torch.autograd.grad(
                        loss, candidate_model.parameters(), create_graph=False
                    )

                    with torch.no_grad():
                        for p, g in zip(candidate_model.parameters(), grads):
                            p -= self.lrate * g

                    del X_batch, y_batch, pred, loss, grads

            candidate_model.eval()
            with torch.no_grad():
                pred = candidate_model(X_train_i)
                eval_loss = self.loss_fn(pred, y_train_i).item()

            candidate_losses.append(eval_loss)
            candidate_states.append(
                {k: v.clone().cpu() for k, v in candidate_model.state_dict().items()}
            )

        # Rank all candidates by loss (ascending) and take top-K
        sorted_idx = sorted(range(S), key=lambda i: candidate_losses[i])
        top_k_idx = sorted_idx[:K]
        top_k_losses = [candidate_losses[i] for i in top_k_idx]
        top_k_states = [candidate_states[i] for i in top_k_idx]

        # Compute aggregation weights
        if self.weighting == 'uniform':
            weights = [1.0 / K] * K
        else:
            # Reward = baseline loss - candidate loss (positive = improvement,
            # negative = candidate made things worse — still rank-meaningful).
            rewards = [baseline_loss - l for l in top_k_losses]
            # Normalise by spread so temperature is scale-invariant regardless
            # of sign or magnitude. Falls back to uniform when all candidates
            # are equivalent (spread ≈ 0).
            r_min, r_max = min(rewards), max(rewards)
            spread = r_max - r_min
            if spread > 1e-8:
                norm_rewards = [(r - r_min) / spread for r in rewards]
            else:
                norm_rewards = [0.0] * K
            exp_r = [math.exp(self.temperature * r) for r in norm_rewards]
            total = sum(exp_r)
            weights = [e / total for e in exp_r] if total > 0 else [1.0 / K] * K

        # Weighted average of top-K state dicts
        avg_state = {
            k: sum(w * s[k] for w, s in zip(weights, top_k_states))
            for k in top_k_states[0].keys()
        }

        best_loss = torch.tensor(top_k_losses[0], device=device)
        return best_loss, avg_state

    def run(self, data):
        self.set_seed()

        X_train, y_train = data["train"]
        X_test, y_test = data["test"]
        cluster_labels = data.get("cluster_labels", None)
        true_weights = data.get("true_weights", None)

        device = self.device
        n_clients = X_train.shape[0]

        X_train = X_train.cpu()
        y_train = y_train.cpu()
        X_test = X_test.cpu()
        y_test = y_test.cpu()

        self.client_models = [self.model_fn().to(device) for _ in range(n_clients)]
        self.loss_history = np.zeros((n_clients, self.R))

        if cluster_labels is not None:
            cluster_labels = torch.as_tensor(cluster_labels, device=device)
        if true_weights is not None:
            true_weights = torch.as_tensor(true_weights, device=device)

        for r in range(self.R):
            if "shift_at" in data and r == data["shift_at"]:
                X_train, y_train = (t.cpu() for t in data["train_shifted"])
                if "test_shifted" in data:
                    X_test, y_test = (t.cpu() for t in data["test_shifted"])

            if self.lrate_decay is not None:
                self.lrate = self.lrate_init * (self.lrate_decay ** r)

            selected_clients = torch.arange(n_clients, device=device)
            selected_list = selected_clients.tolist()

            for i in selected_list:
                self.local_train(self.client_models[i], X_train[i], y_train[i])

            candidate_indices = []
            for i in selected_list:
                pool = torch.tensor([j for j in selected_list if j != i], device=device)
                idx = pool[torch.randperm(len(pool), device=device)[:self.S]]
                candidate_indices.append((i, idx))

            for i, cand_idx in candidate_indices:
                candidates_X = [X_train[j] for j in cand_idx]
                candidates_y = [y_train[j] for j in cand_idx]

                best_loss, avg_params = self.weight_update(
                    self.client_models[i],
                    candidates_X,
                    candidates_y,
                    X_train[i],
                    y_train[i]
                )

                self.loss_history[i, r] = best_loss.detach().cpu().item()
                self.client_models[i].load_state_dict(
                    {k: v.to(device) for k, v in avg_params.items()}
                )

            metrics_sums = {
                name: torch.tensor(0.0, device=device)
                for name in self.metrics.keys()
            }

            for i in range(n_clients):
                test_predictions = None
                for metric_name, metric_fn in self.metrics.items():
                    if metric_name == "MSE_params":
                        if true_weights is None or cluster_labels is None:
                            raise ValueError(
                                "MSE_params requires both true_weights and cluster_labels"
                            )
                        cluster_id = cluster_labels[i]
                        param_tensor = list(self.client_models[i].state_dict().values())[0]
                        metric_value = metric_fn(
                            torch.squeeze(param_tensor), true_weights[cluster_id]
                        )
                    else:
                        if test_predictions is None:
                            X_test_i = X_test[i].to(device)
                            y_test_i = y_test[i].to(device)
                            test_predictions = self.get_predictions(
                                self.client_models[i], X_test_i
                            )
                        metric_value = metric_fn(test_predictions, y_test_i)

                    metrics_sums[metric_name] += metric_value.detach()

                try:
                    del X_test_i, y_test_i, test_predictions
                except NameError:
                    pass

            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = (
                    metrics_sums[metric_name] / n_clients
                )

        return self.client_models
