import torch
import numpy as np
import random
import copy
from torch import nn

class Scaffold:
    def __init__(self, model_fn, loss_fn, metrics, R=50, R_local=0, S=20,
                 lrate=0.01, lrate_decay=None, device='cpu', seed=None):
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

        self.global_model = self.model_fn().to(device)
        self.loss_history = None
        self.metrics_history = {
            name: torch.zeros(self.R, device=self.device)
            for name in metrics.keys()
        }

        # Server control variate (c)
        self.c = [torch.zeros_like(p, device=device) for p in self.global_model.parameters()]

        # Client control variates will be initialized on first use
        self.client_c = None

    # -------------------------------
    # Reproducibility
    # -------------------------------
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

    # -------------------------------
    # Local training with control variate
    # -------------------------------
    def local_train(self, model, X, y, c_global, c_local):
        model.train()
        data_size = X.shape[0]
        batch_size = min(32, data_size)

        for r in range(self.R_local):
            perm = torch.randperm(data_size, device=self.device)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for start in range(0, data_size, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                pred = model(X_batch)
                loss = self.loss_fn(pred, y_batch)

                grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

                with torch.no_grad():
                    for p, g, cg, cl in zip(model.parameters(), grads, c_global, c_local):
                        # SCAFFOLD update: gradient - local_control + global_control
                        p -= self.lrate * (g - cl + cg)

    # -------------------------------
    # Run
    # -------------------------------
    def run(self, data):
        self.set_seed()

        X_train, y_train = data["train"]
        X_val, y_val = data["val"]

        device = self.device
        n_clients = X_train.shape[0]

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        # Initialize client control variates
        if self.client_c is None:
            self.client_c = [
                [torch.zeros_like(p, device=device) for p in self.global_model.parameters()]
                for _ in range(n_clients)
            ]

        self.loss_history = np.zeros((n_clients, self.R))

        for r in range(self.R):
            if self.lrate_decay is not None:
                self.lrate = self.lrate_init * (self.lrate_decay ** r)

            m = min(self.S, n_clients)
            selected_clients = torch.randperm(n_clients, device=device)[:m]

            local_params_list = []
            local_sizes = []
            delta_c_list = []

            for i in selected_clients:
                local_model = copy.deepcopy(self.global_model)
                X_i = X_train[i]
                y_i = y_train[i]

                self.local_train(local_model, X_i, y_i, self.c, self.client_c[i])

                dataset_size = X_i.shape[0]
                local_sizes.append(dataset_size)

                local_params_list.append({
                    k: v.clone()
                    for k, v in local_model.state_dict().items()
                })

                # Compute delta_c_i = (w_global - w_local) / (lr * R_local)
                delta_c = [
                    (g - l) / (self.lrate * self.R_local)
                    for g, l in zip(self.global_model.parameters(), local_model.parameters())
                ]
                delta_c_list.append(delta_c)

            # Aggregate local models
            total_size = sum(local_sizes)
            new_global_state = {}
            for k in self.global_model.state_dict().keys():
                new_global_state[k] = sum(
                    local_params_list[j][k] * (local_sizes[j] / total_size)
                    for j in range(len(local_params_list))
                )
            self.global_model.load_state_dict(new_global_state)

            # Update server control variate
            avg_delta_c = [
                sum(delta_c_list[j][idx] * (local_sizes[j] / total_size)
                    for j in range(len(delta_c_list)))
                for idx in range(len(self.c))
            ]
            self.c = avg_delta_c

            # Update client control variates
            for idx, i in enumerate(selected_clients):
                self.client_c[i] = [
                    cl - cg + avg_dc
                    for cl, cg, avg_dc in zip(self.client_c[i], self.c, delta_c_list[idx])
                ]

            # Training loss after aggregation
            for i in range(n_clients):
                pred = self.get_predictions(self.global_model, X_train[i])
                self.loss_history[i, r] = self.loss_fn(pred, y_train[i]).detach().cpu().item()

            # Evaluate metrics
            metrics_sums = {name: torch.tensor(0.0, device=device) for name in self.metrics.keys()}

            for i in range(n_clients):
                val_predictions = None
                for metric_name, metric_fn in self.metrics.items():
                    if val_predictions is None:
                        val_predictions = self.get_predictions(self.global_model, X_val[i])
                    metric_value = metric_fn(val_predictions, y_val[i])
                    metrics_sums[metric_name] += metric_value.detach()

            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = metrics_sums[metric_name] / n_clients

        return self.global_model