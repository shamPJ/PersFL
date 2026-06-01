import random
import numpy as np
import torch

class FedProx:
    def __init__(
        self,
        model_fn,
        loss_fn,
        metrics,
        R=50,
        R_local=10,
        P=None, # number of participants
        lrate=0.01,
        lrate_decay=None,
        mu=0.01,  # FedProx proximal strength
        device="cpu",
        seed=None
    ):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.metrics = metrics

        self.R = R
        self.R_local = R_local
        self.P = P

        self.lrate_init = lrate
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.mu = mu

        self.device = device
        self.seed = seed

        self.global_model = self.model_fn().to(device)

        self.loss_history = None
        self.metrics_history = {
            name: torch.zeros(self.R, device=self.device)
            for name in metrics.keys()
        }

    # ----------------------------
    # Utilities
    # ----------------------------
    def set_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.device != "cpu":
                torch.cuda.manual_seed_all(self.seed)

    def is_bn_buffer(self, k):
        # not learned params, do not include in proximal term
        return (
            "running_mean" in k
            or "running_var" in k
            or "num_batches_tracked" in k
        )

    # ----------------------------
    # Inference
    # ----------------------------
    def get_predictions(self, model, X):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model(X)
        if was_training:
            model.train()
        return out

    # ----------------------------
    # Client sampling
    # ----------------------------
    def sample_clients(self, n_clients, cluster_labels, active_clusters, device):
        """
        Sample exactly self.P clients from the eligible pool (clients belonging
        to active_clusters).
        """
        eligible = torch.tensor(
            [i for i in range(n_clients) if cluster_labels[i].item() in active_clusters],
            device=device,
        )

        if len(eligible) >= self.P:
            selected = eligible[torch.randperm(len(eligible), device=device)[: self.P]]
        else:
            raise ValueError(
                f"Not enough eligible clients to sample: {len(eligible)} available, but P={self.P} requested. Consider reducing P or increasing n_active_clusters."
            )

        return selected

    # ----------------------------
    # Local training (FedProx SGD)
    # ----------------------------
    def local_train(self, model, X, y, global_state):
        model.train()

        data_size = X.shape[0]
        batch_size = min(32, data_size)

        # build reference (global) parameters for proximal term
        global_params = {
            k: v.detach().clone()
            for k, v in global_state.items()
            if not self.is_bn_buffer(k)
        }

        for _ in range(self.R_local):
            perm = torch.randperm(data_size, device=X.device)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for start in range(0, data_size, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                pred = model(X_batch)
                loss = self.loss_fn(pred, y_batch)

                # ----------------------------
                # FedProx proximal term
                # ----------------------------
                prox = 0.0
                for name, p in model.named_parameters():
                    if name in global_params:
                        prox += torch.sum((p - global_params[name]) ** 2)

                loss = loss + (self.mu / 2.0) * prox

                grads = torch.autograd.grad(
                    loss, model.parameters(), create_graph=False
                )

                with torch.no_grad():
                    for p, g in zip(model.parameters(), grads):
                        p -= self.lrate * g

    # ----------------------------
    # Aggregation (FedAvg-style: average all params including BN stats)
    # ----------------------------
    def aggregate(self, local_states, local_sizes):
        total_size = sum(local_sizes)
        global_state = self.global_model.state_dict()

        new_state = {}

        for k in global_state.keys():
            if "num_batches_tracked" in k:
                new_state[k] = global_state[k]
            else:
                new_state[k] = sum(
                    local_states[j][k] * (local_sizes[j] / total_size)
                    for j in range(len(local_states))
                )

        self.global_model.load_state_dict(new_state)

    # ----------------------------
    # Main loop
    # ----------------------------
    def run(self, data):
        self.set_seed()

        X_train, y_train = data["train"]
        X_test, y_test = data["test"]

        cluster_labels = data.get("cluster_labels", None)
        true_weights = data.get("true_weights", None)

        device = self.device
        n_clients = X_train.shape[0]

        if self.P is None:
            self.P = n_clients

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        if cluster_labels is not None:
            cluster_labels = torch.as_tensor(cluster_labels, device=device)
        if true_weights is not None:
            true_weights = torch.as_tensor(true_weights, device=device)

        m = min(self.P, n_clients)

        self.loss_history = np.zeros((n_clients, self.R))

        for r in range(self.R):
            if "shift_at" in data and r == data["shift_at"]:
                X_train, y_train = (t.to(device) for t in data["train_shifted"])
                if "test_shifted" in data:
                    X_test, y_test = (t.to(device) for t in data["test_shifted"])

            if self.lrate_decay is not None:
                self.lrate = self.lrate_init * (self.lrate_decay ** r)

            selected_clients = torch.randperm(n_clients, device=device)[:m]

            local_states = []
            local_sizes = []

            global_state = self.global_model.state_dict()

            # ------------------------
            # Local training
            # ------------------------
            for i in selected_clients:
                client_idx = i.item()

                model = self.model_fn().to(device)
                model.load_state_dict(global_state)

                X_i = X_train[client_idx]
                y_i = y_train[client_idx]

                self.local_train(model, X_i, y_i, global_state)

                local_states.append(
                    {k: v.detach().clone() for k, v in model.state_dict().items()}
                )
                local_sizes.append(X_i.shape[0])

            # ------------------------
            # Aggregation
            # ------------------------
            self.aggregate(local_states, local_sizes)

            # ------------------------
            # Evaluation
            # ------------------------
            metrics_sums = {
                name: torch.tensor(0.0, device=device)
                for name in self.metrics.keys()
            }
            # Evaluation is done on all clients (not just selected ones) to track global performance across rounds
            for i in range(n_clients):
                model = self.model_fn().to(self.device)
                model.load_state_dict(self.global_model.state_dict())

                pred = self.get_predictions(model, X_train[i])
                self.loss_history[i, r] = self.loss_fn(pred, y_train[i]).detach().cpu().item()

                # validation metrics
                test_predictions = None

                for name, fn in self.metrics.items():
                    if name == "MSE_params":
                        if true_weights is None or cluster_labels is None:
                            raise ValueError(
                                "MSE_params requires true_weights and cluster_labels"
                            )

                        cluster_id = cluster_labels[i]
                        param_tensor = list(model.state_dict().values())[0]

                        metric_value = fn(
                            torch.squeeze(param_tensor),
                            true_weights[cluster_id],
                        )
                    else:
                        if test_predictions is None:
                            test_predictions = self.get_predictions(
                                model,
                                X_test[i],
                            )

                        metric_value = fn(test_predictions, y_test[i])

                    metrics_sums[name] += metric_value.detach()

            for name in self.metrics.keys():
                self.metrics_history[name][r] = metrics_sums[name] / n_clients

        return self.global_model
