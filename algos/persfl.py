import torch
import numpy as np
import random
from torch import nn
from utils.metrics import MSE, MSE_params, accuracy, F1

class PersFL:
    def __init__(self, model_fn, loss_fn, metrics={"MSE_val": MSE}, R=50, R_local=0, S=20, lrate=0.01, device='cpu', seed=None):
        """
        PersFL Algorithm

        Args:
            model_fn: function returning fresh model instance
            loss_fn: differentiable loss function
            R: number of iterations
            S: number of candidate neighbors per client
            lrate: learning rate
            device: 'cpu' or 'cuda'
            kwargs: additional algorithm-specific parameters
        """
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.R = R
        self.R_local = R_local
        self.S = S
        self.lrate = lrate
        self.device = device
        self.seed = seed
        
        # client models initialized later
        self.client_models = None
        self.loss_history = None # training loss as torch tensor (n_clients, R)
        self.metrics_history = {name: torch.zeros(self.R, device=self.device) for name in metrics.keys()}

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
        for _ in range(self.R_local):
            pred = model(X)
            loss = self.loss_fn(pred, y)
            grads = torch.autograd.grad(loss, model.parameters())

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
        X_val, y_val = data["val"] 
        cluster_labels = data.get("cluster_labels", None)
        true_weights = data.get("true_weights", None)

        device = self.device
        n_clients, _, d = X_train.shape

        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

        # Initialize client models
        self.client_models = []
        for _ in range(n_clients):
            model = self.model_fn().to(device)
            self.client_models.append(model)

        self.loss_history = torch.zeros((n_clients, self.R), device=device)

        if cluster_labels is not None:
            cluster_labels = torch.tensor(cluster_labels, device=device)
        if true_weights is not None:
            true_weights = torch.tensor(true_weights, device=device)

        # Main iteration loop
        for r in range(self.R):
            # --- local updates ---
            for i in range(n_clients):
                self.local_train(
                    self.client_models[i],
                    X_train[i],
                    y_train[i]
                )
            # Step 1: sample candidate neighbors (exclude self)
            candidate_indices = []
            for i in range(n_clients):
                pool = torch.cat([
                    torch.arange(0, i, device=device),
                    torch.arange(i+1, n_clients, device=device)
                ])
                idx = pool[torch.randperm(n_clients - 1, device=device)[:self.S]]
                candidate_indices.append(idx)

            # Step 2: compute candidate updates
            all_candidate_params = []
            for i in range(n_clients):
                candidates_X = X_train[candidate_indices[i]]
                candidates_y = y_train[candidate_indices[i]]
                # candidate_params - list of dicts with params
                candidate_params = self.weight_update(self.client_models[i], candidates_X, candidates_y)
                all_candidate_params.append(candidate_params)

            # Step 3: evaluate candidates and select best param. est. error list
            # initialize accumulators per metric
            metrics_sums = {name: torch.tensor(0.0, device=device) for name in self.metrics.keys()}
            for i in range(n_clients):
                losses = self.candidate_losses(self.client_models[i], all_candidate_params[i], X_train[i], y_train[i])
                best_idx = torch.argmin(losses)
                best_w = all_candidate_params[i][best_idx]
                self.client_models[i].load_state_dict(best_w)
                self.loss_history[i, r] = losses[best_idx]

                # -----------------------------
                # Evaluate metrics for this iteration
                # -----------------------------
                val_predictions = None # cache val predictions if needed for multiple metrics
                for metric_name, metric_fn in self.metrics.items():
                    if metric_name == "MSE_params":
                        # only for linear models
                        cluster_id = cluster_labels[i]
                        param_tensor = list(best_w.values())[0]
                        if true_weights is None or cluster_labels is None:
                            raise ValueError("MSE_params requires both true_weights and cluster_labels")
                        # out is scalar tensor
                        metric_value = metric_fn(torch.squeeze(param_tensor), true_weights[cluster_id])
                    else:
                        if val_predictions is None:
                            val_predictions = self.get_predictions(self.client_models[i], X_val[i])
                        metric_value = metric_fn(val_predictions, y_val[i])
                    
                    metrics_sums[metric_name] += metric_value.detach()

            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = metrics_sums[metric_name] / n_clients

        return self.client_models

    # --------------------------------
    # Helper methods
    # --------------------------------
    def weight_update(self, model, X_candidates, y_candidates):
        """
        Compute candidate updates for one client using functional gradient steps.

        Args:
            model: nn.Module, current client model
            X_candidates: Tensor[S, m_i, C, H, W] - inputs for candidate neighbors
            y_candidates: Tensor[S, m_i, ...] - labels for candidate neighbors

        Returns:
            candidate_params: list of S dicts with cloned parameter tensors
        """
        device = self.device
        S = X_candidates.shape[0]
        candidate_params = []

        # --- 1. Store base model parameters as clones ---
        base_params = [p.clone() for p in model.parameters()]
        param_names = list(model.state_dict().keys())

        # Ensure model does not modify running stats for candidates
        model.eval()  # freeze batchnorm/dropout stats

        for i in range(S):
            # Forward pass for candidate i
            # on candidate neighbor's data
            pred = model(X_candidates[i])
            loss = self.loss_fn(pred, y_candidates[i])

            # Compute gradients w.r.t. current parameters (decoupled)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

            # Manual SGD / minibatch update: functional update without touching model
            updated_params = [p - self.lrate * g for p, g in zip(base_params, grads)]

            # Store updated candidate weights as independent cloned state_dict
            candidate_params.append({
                name: p.clone()
                for name, p in zip(param_names, updated_params)
            })

        # Restore model to original parameters
        for p, bp in zip(model.parameters(), base_params):
            p.data.copy_(bp)

        # Return model to train mode if needed for evaluation later
        model.train()

        return candidate_params

    def candidate_losses(self, client_model, candidate_params, X_client, y_client):
        """Evaluate candidates on a single client"""
        device = self.device
        S = len(candidate_params)
        losses = torch.zeros(S, device=device)

        # save client's current params to restore later
        base_state = {k: v.clone() for k, v in client_model.state_dict().items()}

        for i, params in enumerate(candidate_params):
            client_model.load_state_dict(params)
            client_model.eval()
            with torch.no_grad():
                pred = client_model(X_client)
                losses[i] = self.loss_fn(pred.squeeze(), y_client.squeeze())
        client_model.load_state_dict(base_state)
        return losses
