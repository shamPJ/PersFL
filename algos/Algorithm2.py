import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
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
        self.lrate_init = lrate
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

        data_size = X.shape[0]
        batch_size = min(32, data_size)

        for _ in range(self.R_local):
            # Shuffle the dataset at the start of each epoch
            perm = torch.randperm(data_size, device=X.device)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            # Iterate over minibatches
            for start in range(0, data_size, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end].to(self.device, non_blocking=True)
                y_batch = y_shuffled[start:end].to(self.device, non_blocking=True)

                pred = model(X_batch)
                loss = self.loss_fn(pred, y_batch)

                grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

                with torch.no_grad():
                    for i, (p, g) in enumerate(zip(model.parameters(), grads)):
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

        if len(X_train.shape) == 3:
            X_test = torch.randn((100, X_train.shape[2]), device='cpu') # fixed test set for regularization
        elif len(X_train.shape) == 5:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1)

            X_test = torch.randn((1000,3,32,32), device='cpu') * std + mean

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
            metrics_sums = {name: torch.tensor(0.0, device=device) for name in self.metrics.keys()}
            for i in range(n_clients):
                candidates_X = [X_train[j] for j in candidate_indices[i]]
                candidates_y = [y_train[j] for j in candidate_indices[i]]

                best_loss, best_params = self.hypothesis_update(self.client_models[i],
                                            candidates_X,
                                            candidates_y,
                                            X_train[i],
                                            y_train[i],
                                            X_test  
                                        )
                    
                self.loss_history[i, r] = best_loss.detach().cpu().item()
                self.client_models[i].load_state_dict({k: v.to(self.device) for k, v in best_params.items()})

                # -----------------------------
                # Evaluate metrics for this iteration
                # -----------------------------
                val_predictions = None # cache val predictions if needed for multiple metrics
                for metric_name, metric_fn in self.metrics.items():
                    if metric_name == "MSE_params":
                        # only for linear models
                        cluster_id = cluster_labels[i]
                        param_tensor = list(best_params.values())[0]
                        if true_weights is None or cluster_labels is None:
                            raise ValueError("MSE_params requires both true_weights and cluster_labels")
                        # out is scalar tensor
                        metric_value = metric_fn(torch.squeeze(param_tensor), true_weights[cluster_id])
                    else:
                        if val_predictions is None:
                            X_val_i = X_val[i].to(device)
                            y_val_i = y_val[i].to(device)
                            val_predictions = self.get_predictions(self.client_models[i], X_val_i)

                        metric_value = metric_fn(val_predictions, y_val_i)
                        del X_val_i, y_val_i
                    
                    metrics_sums[metric_name] += metric_value.detach()

            for metric_name in self.metrics.keys():
                self.metrics_history[metric_name][r] = metrics_sums[metric_name] / n_clients

        return self.client_models

    # --------------------------------
    # Helper methods
    # --------------------------------
    def hypothesis_update(self, model, X_candidates, y_candidates, X_train_i, y_train_i, X_pub, T=1):
        """
        Model-agnostic update via regularized re-training.

        Args:
            model: current client model (ĥ)
            X_candidates: Tensor [S, ...]
            y_candidates: Tensor [S, ...]
            X_pub: Tensor for prediction regularization

        Returns:
            candidate_models: list of updated model instances
        """
        device = self.device
        # move local data to gpu
        X_train_i = X_train_i.to(device)
        y_train_i = y_train_i.to(device)

        S = len(X_candidates)

        # Ensure model does not modify running stats for candidates
        model.eval()  # freeze batchnorm/dropout stats
        with torch.no_grad():
            base_preds = model(X_pub.to(device))

        candidate_model = self.model_fn().to(device) # reuse model for memory save

        best_loss = torch.tensor(float("inf"), device=device)
        best_params = None
        
        if len(X_train_i.shape) == 3:
            task = "regression"
        elif len(X_train_i.shape) == 5:
             task = "classification"

        for i in range(S):
            # start from clients' model / params
            candidate_model.load_state_dict(model.state_dict())
            candidate_model.train()

            # Simple inner training loop
            for _ in range(1):
                pred = candidate_model(X_candidates[i])
                loss_data = self.loss_fn(pred, y_candidates[i])

                # Prediction regularization
                pred_pub = candidate_model(X_pub.to(device))
                if task == 'regression':
                    loss_reg = torch.mean((pred_pub - base_preds) ** 2)
                elif task == 'classification':
                    p_teacher = F.softmax(base_preds / T, dim=1)
                    p_student = F.log_softmax(pred_pub / T, dim=1)
                    loss_reg = F.kl_div(p_student, p_teacher, reduction='batchmean') * (T**2)

                loss = self.lmbd * loss_data + loss_reg
                loss.backward()

                with torch.no_grad():
                    for p in candidate_model.parameters():
                        p -= self.lrate * p.grad
                        p.grad = None
                
                # cleanup
                del pred, loss

            # eval trained model on client's local data
            with torch.no_grad():
                pred = candidate_model(X_train_i)
                loss = self.loss_fn(pred, y_train_i)
            
            # keep only smallest loss
            if loss < best_loss:
                best_loss = loss
                best_params = {k: v.clone().cpu() for k, v in candidate_model.state_dict().items()}

        # Return model to train mode if needed for evaluation later
        model.train()

        del candidate_model

        return best_loss, best_params
