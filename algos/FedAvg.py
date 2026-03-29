import torch
import numpy as np
import random
import copy
from torch import nn

class FedAvg:
    def __init__(self, model_fn, loss_fn, metrics, R=50, R_local=0, S=20,
        lrate=0.01, momentum=0.0, lrate_decay=None,
        device='cpu', seed=None
    ):
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
    
    def local_train(self, model, X, y, velocity):
        model.train()

        use_momentum = (velocity is not None and self.momentum > 0)

        data_size = X.shape[0]
        batch_size = min(32, data_size)

        for _ in range(self.R_local):
            # batching
            idx = torch.randperm(data_size, device=self.device)[:batch_size]
            X_batch = X[idx]
            y_batch = y[idx]
            
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

        return loss

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
        n_clients = X_train.shape[0]

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        if cluster_labels is not None:
            cluster_labels = torch.tensor(cluster_labels, device=device)
        if true_weights is not None:
            true_weights = torch.tensor(true_weights, device=device)

        # Initialize client models
        self.client_models = [copy.deepcopy(self.global_model) for _ in range(n_clients)]
        self.loss_history = torch.zeros((n_clients, self.R), device=device)

        # Initialize velocities
        if self.momentum > 0:
            self.velocities = []
            for model in self.client_models:
                velocity = [torch.zeros_like(p, device=device) for p in model.parameters()]
                self.velocities.append(velocity)

        # --------------------------------
        # Main loop
        # --------------------------------
        for r in range(self.R):
            # After round 0, add this:
            # if r == 0:
            #     print("=== SHAPE DIAGNOSTICS ===")
            #     print(f"X_train: {X_train.shape}")   # expect [20, 500, 3, 32, 32]
            #     print(f"y_train: {y_train.shape}")   # expect [20, 500]
            #     print(f"X_val:   {X_val.shape}")     # expect [20, N_val, 3, 32, 32]
            #     print(f"y_val:   {y_val.shape}")     # expect [20, N_val]
                
            #     # Check val label distribution per client
            #     for i in range(min(3, n_clients)):
            #         print(f"Client {i} val label distribution: {torch.bincount(y_val[i].long())}")
                
            #     # Sanity check: what does a random model score?
            #     random_model = self.model_fn().to(device)
            #     preds = self.get_predictions(random_model, X_val[0])
            #     pred_labels = preds.argmax(dim=1)
            #     correct = (pred_labels == y_val[0]).float().mean()
            #     print(f"Random model accuracy on client 0 val: {correct.item():.4f}")  # expect ~0.10

            # LR decay
            if self.lrate_decay is not None:
                self.lrate = self.lrate_init * (self.lrate_decay ** r)

            # Step 1: sample clients
            m = min(self.S, n_clients)
            selected_clients = torch.randperm(n_clients, device=device)[:m]

            local_params_list = []
            local_sizes = []

            # Step 2: local training
            for i in selected_clients:
                local_model = copy.deepcopy(self.global_model)
                velocity = self.velocities[i] if self.momentum > 0 else None

                X_i = X_train[i]
                y_i = y_train[i]

                loss = self.local_train(local_model, X_i, y_i, velocity)

                dataset_size = X_i.shape[0]
                local_sizes.append(dataset_size)

                local_params_list.append({
                    k: v.clone()
                    for k, v in local_model.state_dict().items()
                })

                self.loss_history[i, r] = loss.detach()
                print(f"Iter {r}, Client {i}, Loss: {loss.detach().item():.4f}")

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

            # reset velocities to zero after aggregation
            # After Step 4 (broadcast global model)
            if self.momentum > 0:
                for i in range(n_clients):
                    self.velocities[i] = [
                        torch.zeros_like(p) for p in self.client_models[i].parameters()
                    ]

            # --------------------------------
            # Step 5: metric evaluation 
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
                        print(f"{metric_name}: {metric_value.item():.4f}")

                    metrics_sums[metric_name] += metric_value.detach()

            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = (
                    metrics_sums[metric_name] / n_clients
                )

        return self.client_models