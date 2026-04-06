import torch
import numpy as np
import random
from torch import nn
from utils.metrics import MSE, MSE_params, accuracy, F1

class Algorithm1:
    def __init__(self, model_fn, loss_fn, metrics={"MSE_val": MSE}, R=50, R_local=0, S=20, 
                 lrate=0.01, momentum=0.0, lrate_decay=None,
                 device='cpu', seed=None):
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
        self.lrate_init = lrate
        self.lrate = lrate
        self.momentum = momentum
        self.lrate_decay = lrate_decay
        self.velocities = None
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
    
    def local_train(self, model, X, y, velocity):
        """
        velocity: list of tensors for momentum, or None if no momentum
        """
        model.train()

        use_momentum = (velocity is not None and self.momentum > 0)

        data_size = X.shape[0]
        batch_size = min(32, data_size)

        for _ in range(self.R_local):
            # Shuffle the dataset at the start of each epoch
            perm = torch.randperm(data_size, device=self.device)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            # Iterate over minibatches
            for start in range(0, data_size, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                pred = model(X_batch)
                loss = self.loss_fn(pred, y_batch)

                grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

                with torch.no_grad():
                    for i, (p, g) in enumerate(zip(model.parameters(), grads)):
                        if use_momentum:
                            velocity[i] = self.momentum * velocity[i] + g
                            update = velocity[i]
                        else:
                            update = g
                        p -= self.lrate * update
    
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
        
        n_clients = X_train.shape[0]

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        # Initialize client models
        self.client_models = []
        for _ in range(n_clients):
            model = self.model_fn().to(device)
            self.client_models.append(model)

        # Initialize velocities for momentum if needed
        if self.momentum > 0:
            self.velocities = []
            for model in self.client_models:
                velocity = [torch.zeros_like(p, device=device) for p in model.parameters()]
                self.velocities.append(velocity)

        self.loss_history = torch.zeros((n_clients, self.R), device=device)

        if cluster_labels is not None:
            cluster_labels = torch.tensor(cluster_labels, device=device)
        if true_weights is not None:
            true_weights = torch.tensor(true_weights, device=device)

        # Main iteration loop
        for r in range(self.R):
            # lrate decay if specified
            if self.lrate_decay is not None:
                self.lrate = self.lrate_init * (self.lrate_decay ** r)

            # --- Local updates ---
            for i in range(n_clients):
                self.local_train(
                    self.client_models[i],
                    X_train[i],
                    y_train[i],
                    self.velocities[i] if self.momentum > 0 else None
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
                velocity = self.velocities[i] if self.momentum > 0 else None
                # candidate_params - list of dicts with params
                if self.momentum > 0:
                    candidate_params, velocities_on_candidate_set = self.weight_update(self.client_models[i], candidates_X, candidates_y, velocity)
                else:   
                    candidate_params = self.weight_update(self.client_models[i], candidates_X, candidates_y, velocity)
                all_candidate_params.append(candidate_params)

            # Step 3: evaluate candidates and select best param. est. error list
            # initialize accumulators per metric
            metrics_sums = {name: torch.tensor(0.0, device=device) for name in self.metrics.keys()}
            for i in range(n_clients):
                if len(all_candidate_params[i]) == 0:
                    best_w = {k: v.clone() for k, v in self.client_models[i].state_dict().items()}
                    best_idx = 0
                    losses = torch.tensor(
                        [self.loss_fn(self.client_models[i](X_train[i]), y_train[i])],
                        device=device
                    )
                else:
                    losses = self.candidate_losses(self.client_models[i], all_candidate_params[i], X_train[i], y_train[i])
                    best_idx = torch.argmin(losses)
                    best_w = all_candidate_params[i][best_idx]
                    # update client model with best candidate params
                    self.client_models[i].load_state_dict(best_w)
                    # update velocity if using momentum
                    if self.momentum > 0:
                        self.velocities[i] = velocities_on_candidate_set[best_idx]
                self.loss_history[i, r] = losses[best_idx].detach().cpu().numpy()
                # print(f"Iter {r}, Client {i}, Loss: {losses[best_idx].item():.4f}")
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
                        print(f"{metric_name}: {metric_value.item():.4f}")
                    
                    metrics_sums[metric_name] += metric_value.detach()

            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = metrics_sums[metric_name] / n_clients

        return self.client_models

    # --------------------------------
    # Helper methods
    # --------------------------------
    def weight_update(self, model, X_candidates, y_candidates, velocity):
        """
        Compute candidate updates for one client using functional gradient steps.

        Args:
            model: nn.Module, current client model
            X_candidates: Tensor[S, m_i, C, H, W] - inputs for candidate neighbors
            y_candidates: Tensor[S, m_i, ...] - labels for candidate neighbors
            velocity: list of tensors for each parameter if using momentum, or None

        Returns:
            candidate_params: list of S dicts with cloned parameter tensors
        """
        device = self.device
        S = X_candidates.shape[0]
        candidate_params = []

        # --- 1. Store base model parameters as clones ---
        base_params = [p.clone() for p in model.parameters()]
        param_names = [name for name, _ in model.named_parameters()]

        # Ensure model does not modify running stats for candidates
        model.eval()  # freeze batchnorm/dropout stats

        velocities = []
        for i in range(S):
            # create a fresh copy of the model for candidate i
            candidate_model = self.model_fn().to(device)
            candidate_model.load_state_dict(model.state_dict())

            data_size = X_candidates[i].shape[0]
            batch_size = min(32, data_size)

            # reset velocity for this candidate if using momentum
            velocities_candidate = [torch.zeros_like(p) for p in candidate_model.parameters()]

            for r in range(self.R_local):
                # Shuffle the dataset at the start of each epoch
                perm = torch.randperm(data_size, device=self.device)
                X_shuffled = X_candidates[i][perm]
                y_shuffled = y_candidates[i][perm]

                # Iterate over minibatches
                for start in range(0, data_size, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    pred = candidate_model(X_batch)
                    loss = self.loss_fn(pred, y_batch)

                    grads = torch.autograd.grad(loss, candidate_model.parameters(), create_graph=False)

                    with torch.no_grad():
                        for j, (p, g) in enumerate(zip(model.parameters(), grads)):
                            if self.momentum > 0:
                                velocities_candidate[j] = self.momentum * velocities_candidate[j] + g
                                update = velocities_candidate[j]
                            else:
                                update = g
                            p -= self.lrate * update
            
            updated_params = [p.clone() for p in candidate_model.parameters()]
            velocities.append(velocities_candidate)

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

        if self.momentum > 0:
            return candidate_params, velocities
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
                losses[i] = self.loss_fn(pred, y_client)
        client_model.load_state_dict(base_state)
        return losses
