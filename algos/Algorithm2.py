import torch
import numpy as np
import random
from torch import nn
from utils.metrics import MSE, MSE_params, accuracy, F1

class Algorithm2:
    def __init__(self, model_fn, loss_fn, metrics={"MSE_val": MSE}, R=50, R_local=0, S=20, lrate=0.01, lmbd=1, device='cpu', seed=None):
        """
        Algorithm2

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
        self.lmbd = lmbd
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
            loss = self.loss_fn(pred.squeeze(), y.squeeze())
            
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= self.lrate * p.grad
                    p.grad = None
    
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
        X_test = torch.randn((100, d), device=device) # fixed test set for regularization

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
            all_candidate_models = []
            for i in range(n_clients):
                candidates_X = X_train[candidate_indices[i]]
                candidates_y = y_train[candidate_indices[i]]
                # candidate_params - list of dicts with params
                candidate_models = self.hypothesis_update(
                                self.client_models[i],
                                candidates_X,
                                candidates_y,
                                X_test   # acts as test set
                            )
                all_candidate_models.append(candidate_models)

            # Step 3: evaluate candidates and select best param. est. error list
            # initialize accumulators per metric
            metrics_sums = {name: torch.tensor(0.0, device=device) for name in self.metrics.keys()}
            for i in range(n_clients):
                losses = []
                for candidate_model in all_candidate_models[i]:
                    candidate_model.eval()
                    with torch.no_grad():
                        pred = candidate_model(X_train[i])
                        loss = self.loss_fn(pred.squeeze(), y_train[i].squeeze())
                    losses.append(loss)

                losses = torch.stack(losses)
                best_idx = torch.argmin(losses)

                self.client_models[i] = all_candidate_models[i][best_idx]
                self.loss_history[i, r] = losses[best_idx]

                # -----------------------------
                # Evaluate metrics for this iteration
                # -----------------------------
                val_predictions = None # cache val predictions if needed for multiple metrics
                for metric_name, metric_fn in self.metrics.items():
                    if metric_name == "MSE_params":
                        best_w = self.client_models[i].state_dict()
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
    def hypothesis_update(self, model, X_candidates, y_candidates, X_test):
        """
        Model-agnostic update via regularized re-training.

        Args:
            model: current client model (ĥ)
            X_candidates: Tensor [S, ...]
            y_candidates: Tensor [S, ...]
            X_test: Tensor for prediction regularization

        Returns:
            candidate_models: list of updated model instances
        """
        S = X_candidates.shape[0]
        candidate_models = []

        # Cache current model predictions on test set
        base_preds = self.get_predictions(model, X_test).detach()

        base_state = {k: v.clone() for k, v in model.state_dict().items()}
        for i in range(S):
            new_model = self.model_fn().to(self.device)
            new_model.load_state_dict(base_state)
            new_model.train()

            # Simple inner training loop
            for _ in range(1):
                pred = new_model(X_candidates[i])
                loss_data = self.loss_fn(pred.squeeze(), y_candidates[i].squeeze())

                # Prediction regularization
                pred_test = new_model(X_test)
                loss_reg = torch.mean((pred_test - base_preds) ** 2)

                loss = self.lmbd * loss_data + loss_reg
                loss.backward()

                with torch.no_grad():
                    for p in new_model.parameters():
                        p -= self.lrate * p.grad
                        p.grad = None

            candidate_models.append(new_model)

        return candidate_models
