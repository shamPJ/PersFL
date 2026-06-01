import random
import numpy as np
import torch

class IFCA:
    def __init__(
        self,
        model_fn,
        loss_fn,
        metrics,
        R=50,
        R_local=10,
        P=None, # number of participants
        n_clusters=4,
        algo_n_clusters=None, # for passing misspecified n_clusters via command line
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
        self.P = P
        self.lrate_init = lrate
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.device = device
        self.seed = seed
        self.n_clusters = n_clusters if algo_n_clusters is None else algo_n_clusters

        self.cluster_models = [self.model_fn().to(device) for _ in range(self.n_clusters)]
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

    def assign_cluster(self, X_i, y_i):
        """Return index of cluster model with minimum loss on client i's data."""
        best_j, best_loss = 0, float('inf')
        for j, model in enumerate(self.cluster_models):
            model.eval()
            with torch.no_grad():
                loss = self.loss_fn(model(X_i), y_i).item()
            if loss < best_loss:
                best_loss = loss
                best_j = j
        return best_j

    def sample_clients(self, n_clients, cluster_assignments, active_clusters, device):
        """
        Sample exactly self.P clients whose current cluster assignment is in active_clusters.
        cluster_assignments: 1-D tensor of length n_clients with assigned cluster index per client.
        """
        eligible = torch.tensor(
            [i for i in range(n_clients) if cluster_assignments[i].item() in active_clusters],
            device=device,
        )

        if len(eligible) >= self.P:
            selected = eligible[torch.randperm(len(eligible), device=device)[: self.P]]
        else:
            raise ValueError(
                f"Not enough eligible clients to sample: {len(eligible)} available, but P={self.P} requested. Consider reducing P or increasing n_active_clusters."
            )

        return selected

    def local_train(self, model, X, y):
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

    def is_bn_buffer(self, k):
        return "running_mean" in k or "running_var" in k or "num_batches_tracked" in k

    def split_state(self, state_dict):
        bn_state = {}
        non_bn_state = {}

        for k, v in state_dict.items():
            if self.is_bn_buffer(k):
                bn_state[k] = v
            else:
                non_bn_state[k] = v

        return non_bn_state, bn_state

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

            cluster_params = [[] for _ in range(self.n_clusters)]
            cluster_sizes = [[] for _ in range(self.n_clusters)]

            # 1. Assignment step + local training
            for i in selected_clients:
                X_i = X_train[i]
                y_i = y_train[i]

                j = self.assign_cluster(X_i, y_i)

                local_model = self.model_fn().to(device)
                # Initialize local model with current cluster model parameters
                # including BN buffers
                local_model.load_state_dict(self.cluster_models[j].state_dict())

                self.local_train(local_model, X_i, y_i)

                cluster_params[j].append({k: v.clone() for k, v in local_model.state_dict().items()})
                cluster_sizes[j].append(X_i.shape[0])

            # 2. Cluster model update step - weighted average of local models per cluster
            for j in range(self.n_clusters):
                if not cluster_params[j]:
                    continue
                total = sum(cluster_sizes[j])
                new_state = {}
                keys = self.cluster_models[j].state_dict().keys()

                old_state = self.cluster_models[j].state_dict()
                for k in keys:
                    if "num_batches_tracked" in k:
                        new_state[k] = old_state[k]
                    elif self.is_bn_buffer(k):
                        # cluster-wise average of params including BN running stats
                        new_state[k] = sum(
                            cluster_params[j][idx][k] * (cluster_sizes[j][idx] / total)
                            for idx in range(len(cluster_params[j]))
                        )
                    else:
                        new_state[k] = sum(
                            cluster_params[j][idx][k] * (cluster_sizes[j][idx] / total)
                            for idx in range(len(cluster_params[j]))
                        )

                self.cluster_models[j].load_state_dict(new_state)

            # 3. Evaluation step
            for k in range(self.n_clusters):
                self.cluster_models[k].eval()

            metrics_sums = {
                name: torch.tensor(0.0, device=device)
                for name in self.metrics.keys()
            }

            for i in range(n_clients):
                j = self.assign_cluster(X_train[i], y_train[i])
                pred = self.get_predictions(self.cluster_models[j], X_train[i])
                self.loss_history[i, r] = self.loss_fn(pred, y_train[i]).detach().cpu().item()

                test_predictions = None

                for metric_name, metric_fn in self.metrics.items():

                    if metric_name == "MSE_params":
                        if true_weights is None or cluster_labels is None:
                            raise ValueError("MSE_params requires true_weights and cluster_labels")

                        cluster_id = cluster_labels[i]
                        param_tensor = list(self.cluster_models[j].state_dict().values())[0]

                        metric_value = metric_fn(
                            torch.squeeze(param_tensor),
                            true_weights[cluster_id]
                        )

                    else:
                        if test_predictions is None:
                            test_predictions = self.get_predictions(self.cluster_models[j], X_test[i])

                        metric_value = metric_fn(test_predictions, y_test[i])

                    metrics_sums[metric_name] += metric_value.detach()

            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = (
                    metrics_sums[metric_name] / n_clients
                )

        return self.cluster_models
