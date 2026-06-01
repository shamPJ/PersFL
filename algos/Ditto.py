import copy
import random
import numpy as np
import torch


class Ditto:
    def __init__(
        self,
        model_fn,
        loss_fn,
        metrics,
        R=50,
        R_local=10,
        P=None,
        lrate=0.01,
        lrate_decay=None,
        lmbd=0.1,      # proximal regularization for personal models
        device="cpu",
        seed=None,
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
        self.lmbd = lmbd

        self.device = device
        self.seed = seed

        self.global_model = self.model_fn().to(device)
        self.personal_models = None  # list of per-client models, init in run()

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
            if self.device != "cpu":
                torch.cuda.manual_seed_all(self.seed)

    def is_bn_buffer(self, k):
        return (
            "running_mean" in k
            or "running_var" in k
            or "num_batches_tracked" in k
        )

    def get_predictions(self, model, X):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model(X)
        if was_training:
            model.train()
        return out

    def local_train(self, model, X, y):
        """Standard FedAvg local SGD for the global model update."""
        model.train()

        data_size = X.shape[0]
        batch_size = min(32, data_size)

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

                grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

                with torch.no_grad():
                    for p, g in zip(model.parameters(), grads):
                        p -= self.lrate * g

    def personal_train(self, personal_model, X, y, global_state):
        """Proximal gradient update for personal model v_i anchored to global w."""
        personal_model.train()

        global_params = {
            k: v.detach().clone()
            for k, v in global_state.items()
            if not self.is_bn_buffer(k)
        }

        data_size = X.shape[0]
        batch_size = min(32, data_size)

        for _ in range(self.R_local):
            perm = torch.randperm(data_size, device=X.device)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for start in range(0, data_size, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                pred = personal_model(X_batch)
                loss = self.loss_fn(pred, y_batch)

                # proximal term: (lmbd/2) * ||v - w||^2
                prox = 0.0
                for name, p in personal_model.named_parameters():
                    if name in global_params:
                        prox += torch.sum((p - global_params[name]) ** 2)

                loss = loss + (self.lmbd / 2.0) * prox

                grads = torch.autograd.grad(loss, personal_model.parameters(), create_graph=False)

                with torch.no_grad():
                    for p, g in zip(personal_model.parameters(), grads):
                        p -= self.lrate * g

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

        # Initialize personal models as copies of the global model
        self.personal_models = [
            copy.deepcopy(self.global_model) for _ in range(n_clients)
        ]

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

            global_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}

            # ----------------------------
            # Global model update (FedAvg)
            # ----------------------------
            for i in selected_clients:
                client_idx = i.item()

                model = self.model_fn().to(device)
                model.load_state_dict(global_state)

                X_i = X_train[client_idx]
                y_i = y_train[client_idx]

                self.local_train(model, X_i, y_i)

                local_states.append(
                    {k: v.detach().clone() for k, v in model.state_dict().items()}
                )
                local_sizes.append(X_i.shape[0])

            self.aggregate(local_states, local_sizes)

            # ----------------------------
            # Personal model update (proximal)
            # ----------------------------
            for i in selected_clients:
                self.personal_train(
                    self.personal_models[i],
                    X_train[i],
                    y_train[i],
                    global_state,
                )

            # ----------------------------
            # Evaluation on personal models
            # ----------------------------
            metrics_sums = {
                name: torch.tensor(0.0, device=device)
                for name in self.metrics.keys()
            }

            for i in range(n_clients):
                personal_model = self.personal_models[i]

                pred = self.get_predictions(personal_model, X_train[i])
                self.loss_history[i, r] = self.loss_fn(pred, y_train[i]).detach().cpu().item()

                test_predictions = None

                for name, fn in self.metrics.items():
                    if name == "MSE_params":
                        if true_weights is None or cluster_labels is None:
                            raise ValueError(
                                "MSE_params requires true_weights and cluster_labels"
                            )
                        cluster_id = cluster_labels[i]
                        param_tensor = list(personal_model.state_dict().values())[0]
                        metric_value = fn(
                            torch.squeeze(param_tensor),
                            true_weights[cluster_id],
                        )
                    else:
                        if test_predictions is None:
                            test_predictions = self.get_predictions(personal_model, X_test[i])
                        metric_value = fn(test_predictions, y_test[i])

                    metrics_sums[name] += metric_value.detach()

            for name in self.metrics.keys():
                self.metrics_history[name][r] = metrics_sums[name] / n_clients

        return self.personal_models
