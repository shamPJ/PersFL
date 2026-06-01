import torch
import numpy as np
import random
from torch import nn
from utils.metrics import MSE, MSE_params, accuracy, F1

class Algorithm1:
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
            device='cpu', 
            seed=None
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
        self.device = device
        self.seed = seed
        
        # client models initialized later
        self.client_models = None
        self.candidate_model = self.model_fn().to(self.device)
        self.loss_history = None # training loss as torch tensor (n_clients, R)
        self.metrics_history = {name: torch.zeros(self.R, device=self.device) for name in metrics.keys()}
        self.theory_log = None   # populated only for synthetic data (when noise_scale provided)

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

    def local_train(self, model, X, y):

        model.train()

        data_size = X.shape[0]
        batch_size = min(32, data_size)
        
        X_i = X.to(self.device)
        y_i = y.to(self.device)

        for r in range(self.R_local): 
            # Shuffle the dataset at the start of each epoch
            perm = torch.randperm(data_size, device=X_i.device)
            X_shuffled = X_i[perm]
            y_shuffled = y_i[perm]

            # Iterate over minibatches
            for start in range(0, data_size, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end].to(device=X_i.device, non_blocking=True)
                y_batch = y_shuffled[start:end].to(device=X_i.device, non_blocking=True)

                pred = model(X_batch)
                loss = self.loss_fn(pred, y_batch)

                grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

                with torch.no_grad():
                    for p, g in zip(model.parameters(), grads):
                            p -= self.lrate * g
    
    # --------------------------------
    # Run method: 
    # --------------------------------
    def run(self, data):
        """
        Run PersFL algorithm

        Args:
            data: tuple (X_train, y_train), X_train: (n_clients, m_i, d)
            cluster_labels: optional, cluster assignment for each client
            true_weights: optional, for computing deviation from true weights

        Returns:
            client_models: list of nn.Module
        """
        self.set_seed()

        X_train, y_train = data["train"]
        X_test, y_test = data["test"]
        cluster_labels = data.get("cluster_labels", None)
        # only available for synthetic data
        true_weights = data.get("true_weights", None)
        noise_scale = data.get("noise_scale", None)
        sigma_sq = float(noise_scale) ** 2 if noise_scale is not None else None

        device = self.device

        n_clients = X_train.shape[0]
        if self.S > n_clients - 1:
            raise ValueError(f"S={self.S} exceeds maximum candidate pool size n-1={n_clients - 1}")

        # keeping data on CPU due to OOM errors
        X_train = X_train.cpu()
        y_train = y_train.cpu()
        X_test = X_test.cpu()
        y_test = y_test.cpu()

        # Initialize client models
        self.client_models = []
        for _ in range(n_clients):
            model = self.model_fn().to(device)
            self.client_models.append(model)

        # array to store loss of the "best" candidate
        self.loss_history = np.zeros((n_clients, self.R))

        if cluster_labels is not None:
            cluster_labels = torch.as_tensor(cluster_labels, device=device)
        if true_weights is not None:
            true_weights = torch.as_tensor(true_weights, device=device)

        # Theory log: only for synthetic data (true_weights + noise_scale both present)
        # needed for testing corollary "Convergence regimes"
        # Analytic formula for true loss for syn.data exp's 
        # L_1 = ||w - w^c||^2 + sigma^2, where w is model params and w^c is true cluster weight and sigma^2 is label noise.
        # Mis-selection noise: \xi_{i,r} = est_L1- true_L1 
        # Suboptimality gap: \delta_{i,r} = L_1(w_r^i) - L_1(w_r^{i^*})
        if true_weights is not None and sigma_sq is not None:
            self.theory_log = {
                "xi":                       np.zeros((n_clients, self.R, self.S)),
                "true_loss_cand":           np.zeros((n_clients, self.R, self.S)),
                "candidate_same_cluster":   np.zeros((n_clients, self.R, self.S), dtype=bool),
                "selected_candidate_idx":   np.zeros((n_clients, self.R), dtype=int),
            }
        else:
            self.theory_log = None

        # Main iteration loop
        for r in range(self.R):
            if "shift_at" in data and r == data["shift_at"]:
                X_train, y_train = (t.cpu() for t in data["train_shifted"])
                if "test_shifted" in data:
                    X_test, y_test = (t.cpu() for t in data["test_shifted"])

            # lrate decay if specified
            if self.lrate_decay is not None:
                self.lrate = self.lrate_init * (self.lrate_decay ** r)

            # local updates
            for i in range(n_clients):
                self.local_train(self.client_models[i], X_train[i], y_train[i])

            # candidate sampling and updates
            for i in range(n_clients):
                pool = torch.tensor([j for j in range(n_clients) if j != i], device=device)
                cand_idx = pool[torch.randperm(len(pool), device=device)[:self.S]]

                candidates_X = [X_train[j] for j in cand_idx]
                candidates_y = [y_train[j] for j in cand_idx]

                cluster_labels_S = cluster_labels[cand_idx] if cluster_labels is not None else None
                cluster_label_client = cluster_labels[i] if cluster_labels is not None else None

                tw_i = true_weights[cluster_label_client] if self.theory_log is not None else None

                best_loss, best_params, theory_data = self.weight_update(
                    self.client_models[i],
                    candidates_X,
                    candidates_y,
                    X_train[i],
                    y_train[i],
                    cluster_labels_S,
                    cluster_label_client,
                    cand_idx=cand_idx,
                    true_weight_client=tw_i,
                    sigma_sq=sigma_sq,
                )

                self.loss_history[i, r] = best_loss.detach().cpu().item()
                self.client_models[i].load_state_dict({k: v.to(self.device) for k, v in best_params.items()})

                if theory_data is not None:
                    self.theory_log["xi"]                    [i, r] = theory_data["xi"]
                    self.theory_log["true_loss_cand"]        [i, r] = theory_data["true_loss_cand"]
                    self.theory_log["candidate_same_cluster"][i, r] = theory_data["candidate_same_cluster"]
                    self.theory_log["selected_candidate_idx"][i, r] = theory_data["selected_candidate_idx"]

            # -----------------------------
            # Evaluate metrics
            # -----------------------------
            metrics_sums = {name: torch.tensor(0.0, device=device) for name in self.metrics.keys()}
            for i in range(n_clients):
                test_predictions = None
                for metric_name, metric_fn in self.metrics.items():
                    if metric_name == "MSE_params":
                        if true_weights is None or cluster_labels is None:
                            raise ValueError("MSE_params requires both true_weights and cluster_labels")
                        cluster_id = cluster_labels[i]
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

    # --------------------------------
    # Helper methods
    # --------------------------------
    def load_bn_state(self, model, bn_state):
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.copy_(bn_state[name][0])
                m.running_var.copy_(bn_state[name][1])
                
    def weight_update(self, model, X_candidates, y_candidates, X_train_i, y_train_i, cluster_labels_S, cluster_label_client,
                      cand_idx=None, true_weight_client=None, sigma_sq=None):
        """
        Compute candidate updates for one client using mini-batch GD.

        Args:
            model: nn.Module, current client model
            X_candidates: Tensor[S, m_i, C, H, W] - inputs for candidate neighbors
            y_candidates: Tensor[S, m_i, ...] - labels for candidate neighbors

        Returns:
            
        """
        device = self.device
        # move local data to GPU
        X_train_i = X_train_i.to(device)
        y_train_i = y_train_i.to(device)

        # Ensure model does not modify running stats for candidates
        base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        client_bn_stats = {
            name: (mod.running_mean.clone(), mod.running_var.clone())
            for name, mod in model.named_modules()
            if isinstance(mod, nn.BatchNorm2d)
        }
        
        candidate_model = self.candidate_model # reuse model for memory save
        
        best_loss = torch.tensor(float("inf"), device=device)
        best_params = None
        best_s = -1

        do_log = true_weight_client is not None
        if do_log:
            # arrays for logging true loss val's 
            # L_1 = ||w - w^c||^2 + sigma^2
            cand_true = np.empty(self.S)
            cand_est  = np.empty(self.S)

        for s in range(self.S):
            # start from clients' model / params
            candidate_model.load_state_dict(base_state, strict=True)
            # explicitly load client's BN stats for candidate update
            self.load_bn_state(candidate_model, client_bn_stats)
            candidate_model.train()

            data_size = X_candidates[s].shape[0]
            batch_size = min(32, data_size)

            X_i = X_candidates[s].to(device)
            y_i = y_candidates[s].to(device)

            # train on candidates data
            for _ in range(self.R_local):
                # Shuffle the dataset at the start of each epoch
                perm = torch.randperm(data_size, device=X_i.device)
                X_shuffled = X_i[perm]
                y_shuffled = y_i[perm]

                # Iterate over minibatches
                for start in range(0, data_size, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    pred = candidate_model(X_batch)
                    loss = self.loss_fn(pred, y_batch)

                    grads = torch.autograd.grad(loss, candidate_model.parameters(), create_graph=False)

                    with torch.no_grad():
                        for p, g in zip(candidate_model.parameters(), grads):
                            p -= self.lrate * g

                # cleanup
                del X_batch, y_batch, pred, loss, grads

            # evaluate on client's own training data
            candidate_model.eval()
            with torch.no_grad():
                pred = candidate_model(X_train_i)
                loss = self.loss_fn(pred, y_train_i)

            if do_log:
                param_s = torch.cat([p.detach().reshape(-1) for p in candidate_model.parameters()])
                cand_true[s] = ((param_s - true_weight_client) ** 2).sum().item() + sigma_sq
                cand_est[s]  = loss.item()

            # keep only smallest loss
            if loss < best_loss:
                best_loss = loss
                best_params = {k: v.clone().cpu() for k, v in candidate_model.state_dict().items()}
                best_s = s

        if do_log:
            same_cluster = (cluster_labels_S == cluster_label_client).cpu().numpy().astype(bool)
            theory_data = {
                "xi":                     cand_est - cand_true,
                "true_loss_cand":         cand_true,
                "candidate_same_cluster":   same_cluster,
                "selected_candidate_idx":   cand_idx[best_s].item(),
            }
        else:
            theory_data = None

        return best_loss, best_params, theory_data
