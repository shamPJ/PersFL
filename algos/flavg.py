import torch
import numpy as np
import random
import copy
from torch import nn

class FedAvg:
    def __init__(
        self,
        model_fn,
        loss_fn,
        metrics,
        R=50,
        S=20,
        lrate=0.01,
        device='cpu',
        seed=None
    ):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.R = R
        self.S = S
        self.lrate = lrate
        self.device = device
        self.seed = seed

        self.global_model = self.model_fn().to(device)
        self.client_models = None
        self.loss_history = None
        self.metrics_history = {
            name: torch.zeros(self.R, device=self.device)
            for name in metrics.keys()
        }

    # --------------------------------
    # Reproducibility
    # --------------------------------
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

    # --------------------------------
    # Run
    # --------------------------------
    def run(self, data):
        self.set_seed()

        X_train, y_train = data["train"]
        X_val, y_val = data["val"]
        cluster_labels = data.get("cluster_labels", None)
        true_weights = data.get("true_weights", None)

        device = self.device
        n_clients, _, _ = X_train.shape

        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

        if cluster_labels is not None:
            cluster_labels = torch.tensor(cluster_labels, device=device)
        if true_weights is not None:
            true_weights = torch.tensor(true_weights, device=device)

        # Initialize client models
        self.client_models = [copy.deepcopy(self.global_model) for _ in range(n_clients)]
        self.loss_history = torch.zeros((n_clients, self.R), device=device)

        # --------------------------------
        # Main loop
        # --------------------------------
        for r in range(self.R):

            # Step 1: sample clients
            m = min(self.S, n_clients)
            selected_clients = torch.randperm(n_clients, device=device)[:m]

            local_params_list = []
            local_sizes = []

            # Step 2: local training
            for i in selected_clients:
                local_model = copy.deepcopy(self.global_model)
                local_model.train()

                optimizer = torch.optim.SGD(local_model.parameters(), lr=self.lrate)

                X_i = X_train[i]
                y_i = y_train[i]

                dataset_size = X_i.shape[0]
                local_sizes.append(dataset_size)

                optimizer.zero_grad()
                pred = local_model(X_i)
                loss = self.loss_fn(pred.squeeze(), y_i.squeeze())
                loss.backward()
                optimizer.step()

                local_params_list.append({
                    k: v.clone()
                    for k, v in local_model.state_dict().items()
                })

                self.loss_history[i, r] = loss.detach()

            # Step 3: aggregation
            total_size = sum(local_sizes)
            new_global_state = {}

            for k in self.global_model.state_dict().keys():
                new_global_state[k] = sum(
                    local_params_list[j][k] * (local_sizes[j] / total_size)
                    for j in range(len(local_params_list))
                )

            self.global_model.load_state_dict(new_global_state)

            # Step 4: broadcast global model
            for i in range(n_clients):
                self.client_models[i].load_state_dict(self.global_model.state_dict())

            # --------------------------------
            # Step 5: metric evaluation (aligned with PersFL)
            # --------------------------------
            metrics_sums = {
                name: torch.tensor(0.0, device=device)
                for name in self.metrics.keys()
            }

            for i in range(n_clients):
                val_predictions = None

                for metric_name, metric_fn in self.metrics.items():

                    if metric_name == "MSE_params":
                        if true_weights is None or cluster_labels is None:
                            raise ValueError("MSE_params requires true_weights and cluster_labels")

                        cluster_id = cluster_labels[i]
                        param_tensor = list(self.client_models[i].state_dict().values())[0]

                        metric_value = metric_fn(
                            torch.squeeze(param_tensor),
                            true_weights[cluster_id]
                        )

                    else:
                        if val_predictions is None:
                            val_predictions = self.get_predictions(
                                self.client_models[i],
                                X_val[i]
                            )

                        metric_value = metric_fn(val_predictions, y_val[i])

                    metrics_sums[metric_name] += metric_value.detach()

            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = (
                    metrics_sums[metric_name] / n_clients
                )

        return self.client_models