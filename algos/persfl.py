import torch
import copy
from torch import nn

class PersFL:
    def __init__(self, model_fn, loss_fn, R=50, S=20, lrate=0.01, optimizer_cls=torch.optim.SGD, device='cpu', **kwargs):
        """
        PersFL Algorithm

        Args:
            model_fn: function returning fresh model instance
            loss_fn: differentiable loss function
            R: number of iterations
            S: number of candidate neighbors per client
            lrate: learning rate
            optimizer_cls: torch optimizer class
            device: 'cpu' or 'cuda'
            kwargs: additional algorithm-specific parameters
        """
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.R = R
        self.S = S
        self.lrate = lrate
        self.optimizer_cls = optimizer_cls
        self.device = device
        self.kwargs = kwargs

        # client models initialized later
        self.client_models = None
        self.loss_history = None # training loss (n_clients, R)
        self.MSE = None # param est. error ||true_param - est.param||_2^2 aver. across clients (R,)

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

        X_train, y_train = data["train"]
        cluster_labels = data.get("cluster_labels", None)
        true_weights = data.get("true_weights", None)

        device = self.device
        n_clients, _, d = X_train.shape

        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)

        # Initialize client models
        self.client_models = []
        for _ in range(n_clients):
            model = self.model_fn().to(device)
            nn.init.zeros_(model.linear.weight)
            self.client_models.append(model)

        self.loss_history = torch.zeros((n_clients, self.R), device=device)
        self.MSE = torch.zeros(self.R, device=device)

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        if cluster_labels is not None:
            cluster_labels = torch.tensor(cluster_labels, device=device)
        if true_weights is not None:
            true_weights = torch.tensor(true_weights, device=device)

        # Main iteration loop
        for r in range(self.R):
            # Step 1: sample candidate neighbors
            # candidate_indices = torch.randint(0, n_clients-1, (n_clients, self.S), device=device) # (n_clients, S)
            candidate_list = [torch.randperm(n_clients-1, device=device)[:self.S] for _ in range (n_clients)]                   
            candidate_indices = torch.stack(candidate_list, dim=0)
            # shift indices to exclude node itself
            candidate_indices = candidate_indices + (candidate_indices >= torch.arange(n_clients, device=device).unsqueeze(1)).long()

            # Step 2: compute candidate updates
            all_candidate_params = []
            for i in range(n_clients):
                candidates_X = X_train[candidate_indices[i]]
                candidates_y = y_train[candidate_indices[i]]
                # candidate_params - list of dicts with params
                candidate_params = self.weight_update(self.client_models[i], candidates_X, candidates_y)
                all_candidate_params.append(candidate_params)

            # Step 3: evaluate candidates and select best
            mse_list = [] # param. est. error list
            for i in range(n_clients):
                losses = self.candidate_losses(self.client_models[i], all_candidate_params[i], X_train[i], y_train[i])
                best_idx = torch.argmin(losses)
                best_w = all_candidate_params[i][best_idx]
                self.client_models[i].load_state_dict(best_w)
                self.loss_history[i, r] = losses[best_idx]

                if cluster_labels is not None and true_weights is not None:
                    cluster_id = cluster_labels[i]
                    param_tensor = list(best_w.values())[0]
                    mse_list.append((true_weights[cluster_id] - torch.squeeze(param_tensor)) ** 2)  

            if cluster_labels is not None and true_weights is not None:
                self.MSE[r] = torch.mean(torch.stack(mse_list))

        return self.client_models

    # --------------------------------
    # Helper methods
    # --------------------------------
    def weight_update(self, model, X_candidates, y_candidates):
        """Compute candidate updates for one client"""
        S = X_candidates.shape[0]
        device = self.device
        candidate_params = []

        for i in range(S):
            # send params to neighbours
            model_candidate = copy.deepcopy(model).to(device)
            optimizer = self.optimizer_cls(model_candidate.parameters(), lr=self.lrate)
            model_candidate.train()

            optimizer.zero_grad()
            X = X_candidates[i]
            y = y_candidates[i]
            pred = model_candidate(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            # for p in model_candidate.parameters():
            #     print(p.grad)
            optimizer.step()

            candidate_params.append({name: param.detach().clone() for name, param in model_candidate.named_parameters()})

        return candidate_params
    
# def weight_update_batched(self, model, X_candidates, y_candidates):
    # """
    # General batched update for ANY model (MLP, CNN, etc.)
    # X_candidates: (S, ...)
    # y_candidates: (S, ...)
    # """
    # device = self.device
    # S = X_candidates.shape[0]

    # # Extract base params & buffers
    # base_params = dict(model.named_parameters())
    # base_buffers = dict(model.named_buffers())

    # # Stack parameters: (S, ...)
    # params = {
    #     k: v.detach().clone().unsqueeze(0).repeat(S, *([1] * v.dim()))
    #     for k, v in base_params.items()
    # }

    # buffers = {
    #     k: v.unsqueeze(0).repeat(S, *([1] * v.dim()))
    #     for k, v in base_buffers.items()
    # }

    # # Enable gradients
    # for p in params.values():
    #     p.requires_grad_(True)

    # # Vectorized loss over S candidates
    # batched_loss = vmap(
    #     lambda p, b, X, y: single_candidate_loss(
    #         p, b, model, X, y, self.loss_fn
    #     )
    # )

    # losses = batched_loss(params, buffers, X_candidates, y_candidates)
    # losses.sum().backward()

    # # SGD update
    # with torch.no_grad():
    #     for k in params:
    #         params[k] -= self.lrate * params[k].grad

    # # Return list of parameter dicts
    # candidate_params = [
    #     {k: params[k][i].detach().clone() for k in params}
    #     for i in range(S)
    # ]

    # return candidate_params

    def candidate_losses(self, client_model, candidate_params, X_client, y_client):
        """Evaluate candidates on a single client"""
        device = self.device
        S = len(candidate_params)
        losses = torch.zeros(S, device=device)

        X_client = X_client.to(device)
        y_client = y_client.to(device)

        for i, params in enumerate(candidate_params):
            model = copy.deepcopy(client_model)
            model.load_state_dict(params)
            model.eval()
            with torch.no_grad():
                pred = model(X_client)
                losses[i] = self.loss_fn(pred, y_client)
        return losses
