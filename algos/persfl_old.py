import torch
import copy

def weight_update(model, X_candidates, y_candidates,  optimizer_cls, eta, loss_fn):
    """
    Compute weight update for all candidate datasets in subset S, given current model params of node i.
    
    Args:
        model: nn.Module, model on client i (reference node)
        X_candidates: (S, m_i, d) tensor
        y_candidates: (S, m_i, *)
        optimizer: torch.optim.Optimizer class (e.g., torch.optim.Adam)
        eta: learning rate
        loss_fn: differentiable loss function
    
    Returns:
        candidate_params: list of dicts, each dict is updated parameters for a candidate
    """
    S = X_candidates.shape[0]
    device = X_candidates.device
    
    candidate_params = []
    
    for i in range(S):
        # Clone model
        model_candidate = copy.deepcopy(model).to(device)
        optimizer = optimizer_cls(model_candidate.parameters(), lr=eta)
        model_candidate.train()
        
        # Forward + backward
        optimizer.zero_grad()

        X = X_candidates[i]
        y = y_candidates[i]
        pred = model_candidate(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()

        updated_params = {name: param.detach().clone() for name, param in model_candidate.named_parameters()}
        candidate_params.append(updated_params)
    return  candidate_params


def candidate_losses(client_model, candidate_params, X_client, y_client, loss_fn):
    """
    Evaluation of all candidate models on a single client.
    
    Args:
        client_model: nn.Module template
        candidate_params: list of size S with dicts (model params)
        X_client: (m_i, d)
        y_client: (m_i, *)
        loss_fn: loss function
    
    Returns:
        losses: tensor (S,)
    """
    device = X_client.device
    S = len(candidate_params)
    losses = torch.zeros(S, device=device)
    
    X_client = X_client.to(device)
    y_client = y_client.to(device)
    
    # Evaluate all candidates
    for i, params in enumerate(candidate_params):
        model = copy.deepcopy(client_model)
        model.load_state_dict(params) # Copy parameters and buffers from state_dict into this module and its descendants.
        model.eval()
        
        with torch.no_grad():
            pred = model(X_client)
            losses[i] = loss_fn(pred, y_client).mean()
    
    return losses


def PersFl(ds_train, model_template, loss_fn, cluster_labels=None, true_weights= None, optimizer_cls=torch.optim.SGD, eta=0.1, R=50, S=5, device='cpu'):
    """
    PersFL for all clients.
    
    Args:
        ds_train: tuple (X_train, y_train), X_train: (n_clients, m_i, d)
        model_template: nn.Module
        loss_fn: differentiable loss function
        eta: learning rate
        R: number of iterations
        S: number of candidates per client
        device: 'cpu' or 'cuda'
    
    Returns:
        client_models: list of nn.Module, final models for each client
        loss_history: tensor (n_clients, R)
    """
    X_train, y_train = ds_train
    n_clients = X_train.shape[0]
    
    # Initialize models for all clients
    client_models = [copy.deepcopy(model_template).to(device) for _ in range(n_clients)]
    loss_history = torch.zeros((n_clients, R), device=device)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # Convert cluster_labels to tensor if needed
    if cluster_labels is not None:
        cluster_labels = torch.tensor(cluster_labels, device=device)
    
    MSE = torch.zeros(R, device=device) # deviation from true params
    if true_weights is not None: 
        true_weights = torch.tensor(true_weights, device=device)
    
    for r in range(R):
        # Step 1: sample candidate subsets for all clients
        candidate_indices = torch.randint(0, n_clients-1, (n_clients, S), device=device)
        # shift indices to exclude node itself
        candidate_indices = candidate_indices + (candidate_indices >= torch.arange(n_clients, device=device).unsqueeze(1)).long()

        # Step 2: compute all candidate updates
        all_candidate_params = []
        for i in range(n_clients):
            candidates_X = X_train[candidate_indices[i]]  # (S, m_i, d)
            candidates_y = y_train[candidate_indices[i]]  # (S, m_i, *)
            candidate_params = weight_update(client_models[i], candidates_X, candidates_y, optimizer_cls, eta, loss_fn)
            all_candidate_params.append(candidate_params)
        
        # Step 3: evaluate candidates and select best per client
        mse_list = []
        for i in range(n_clients):
            losses = candidate_losses(client_models[i], all_candidate_params[i], X_train[i], y_train[i], loss_fn)
            best_idx = torch.argmin(losses)
            best_w = all_candidate_params[i][best_idx]
            client_models[i].load_state_dict(best_w)
            loss_history[i, r] = losses[best_idx]

            # Get cluster IDs if using clusters
            if cluster_labels is not None:
                cluster_id = cluster_labels[i]
                param_tensor = list(best_w.values())[0]
                mse_list.append( (true_weights[cluster_id] - param_tensor)**2 )
        
        
        if cluster_labels is not None:
            MSE[r] = torch.mean(torch.stack(mse_list))

    return client_models, loss_history, MSE