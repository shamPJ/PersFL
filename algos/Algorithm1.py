import torch
import numpy as np
import random
from torch import nn
from utils.metrics import MSE, MSE_params, accuracy, F1

class Algorithm1:
    def __init__(self, model_fn, loss_fn, metrics={"MSE_val": MSE}, R=50, R_local=0, S=20, 
                 lrate=0.01, lrate_decay=None,
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
        self.lrate_decay = lrate_decay
        self.device = device
        self.seed = seed
        
        # client models initialized later
        self.client_models = None
        self.candidate_model = self.model_fn().to(self.device)
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

        data_size = X.shape[0]
        batch_size = min(32, data_size)

        for r in range(self.R_local): # note this is n.o. local gradient steps, not epochs
            idx = torch.randint(0, data_size, (batch_size,), device=self.device)

            X_batch = X[idx]
            y_batch = y[idx]

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
        X_val, y_val = data["val"] 
        cluster_labels = data.get("cluster_labels", None)
        true_weights = data.get("true_weights", None)

        device = self.device
        
        n_clients = X_train.shape[0]

        X_train = X_train.cpu()
        y_train = y_train.cpu()
        X_val = X_val.cpu()
        y_val = y_val.cpu()

        # Initialize client models
        self.client_models = []
        for _ in range(n_clients):
            model = self.model_fn().to(device)
            self.client_models.append(model)

        self.loss_history = np.zeros((n_clients, self.R))

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
                    y_train[i]
                )

            # Step 1: sample candidate neighbors (exclude self)
            candidate_indices = []
            for i in range(n_clients):
                pool = torch.cat([
                    torch.arange(0, i),
                    torch.arange(i+1, n_clients)
                ])
                idx = pool[torch.randperm(n_clients - 1)[:self.S]]
                candidate_indices.append(idx)

            # Step 2: compute candidate updates
            # initialize accumulators per metric
            metrics_sums = {name: torch.tensor(0.0, device=device) for name in self.metrics.keys()}
            for i in range(n_clients):
                candidates_X = [X_train[j] for j in candidate_indices[i]]
                candidates_y = [y_train[j] for j in candidate_indices[i]]
                
                best_loss, best_params = self.weight_update(self.client_models[i], 
                                                            candidates_X, 
                                                            candidates_y,
                                                            X_train[i],
                                                            y_train[i]
                                                            )

                self.loss_history[i, r] = best_loss.detach().cpu().item()
                best_params_filtered = {
                    k: v for k, v in best_params.items()
                    if not any(bn_key in k for bn_key in [
                        'running_mean', 'running_var', 'num_batches_tracked'
                    ])
                }

                # Load filtered params, keeping client's own BN stats intact
                state = self.client_models[i].state_dict()
                state.update({k: v.to(self.device) for k, v in best_params_filtered.items()})
                self.client_models[i].load_state_dict(state)
                # self.client_models[i].load_state_dict({k: v.to(self.device) for k, v in best_params.items()})

                # -----------------------------
                # Evaluate metrics for this iteration
                # -----------------------------
                val_predictions = None # cache val predictions if needed for multiple metrics
                for metric_name, metric_fn in self.metrics.items():
                    if metric_name == "MSE_params":
                        # only for linear models
                        if true_weights is None or cluster_labels is None:
                            raise ValueError("MSE_params requires both true_weights and cluster_labels")
                        cluster_id = cluster_labels[i]
                        param_tensor = list(best_params.values())[0]
    
                        # out is scalar tensor
                        metric_value = metric_fn(torch.squeeze(param_tensor), true_weights[cluster_id])
                    else:
                        if val_predictions is None:
                            X_val_i = X_val[i].to(device)
                            y_val_i = y_val[i].to(device)
                            val_predictions = self.get_predictions(self.client_models[i], X_val_i)

                        metric_value = metric_fn(val_predictions, y_val_i)

                    metrics_sums[metric_name] += metric_value.detach()

                try:
                    del X_val_i, y_val_i, val_predictions
                except NameError:
                    pass

            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = metrics_sums[metric_name] / n_clients

        return self.client_models

    # --------------------------------
    # Helper methods
    # --------------------------------
    def weight_update(self, model, X_candidates, y_candidates, X_train_i, y_train_i):
        """
        Compute candidate updates for one client using functional gradient steps.

        Args:
            model: nn.Module, current client model
            X_candidates: Tensor[S, m_i, C, H, W] - inputs for candidate neighbors
            y_candidates: Tensor[S, m_i, ...] - labels for candidate neighbors

        Returns:
            
        """
        device = self.device
        # move local data to gpu
        X_train_i = X_train_i.to(device)
        y_train_i = y_train_i.to(device)

        S = len(X_candidates)

        # Ensure model does not modify running stats for candidates
        base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        candidate_model = self.candidate_model # reuse model for memory save
        
        best_loss = torch.tensor(float("inf"), device=device)
        best_params = None

        for i in range(S):
            # start from clients' model / params
            candidate_model.load_state_dict(base_state, strict=True)
            candidate_model.train()
            for m in candidate_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            data_size = X_candidates[i].shape[0]
            batch_size = min(32, data_size)

            X_i = X_candidates[i].to(device)
            y_i = y_candidates[i].to(device)

            # train on candidates data
            for _ in range(self.R_local):
                idx = torch.randint(0, data_size, (batch_size,), device=self.device)

                X_batch = X_i[idx]
                y_batch = y_i[idx]

                pred = candidate_model(X_batch)
                loss = self.loss_fn(pred, y_batch)

                grads = torch.autograd.grad(loss, candidate_model.parameters(), create_graph=False)

                with torch.no_grad():
                    for p, g in zip(candidate_model.parameters(), grads):
                        p -= self.lrate * g
                
                # cleanup
                del X_batch, y_batch, pred, loss, grads
            
            # eval trained model on client's local data
            with torch.no_grad():
                pred = candidate_model(X_train_i)
                loss = self.loss_fn(pred, y_train_i)
            
            # keep only smallest loss
            if loss < best_loss:
                best_loss = loss
                best_params = {k: v.clone().cpu() for k, v in candidate_model.state_dict().items()}

        return best_loss, best_params
