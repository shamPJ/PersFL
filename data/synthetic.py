import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_data(
        n_clusters,
        n_clients,
        n_samples,
        n_samples_test,
        n_features,
        noise_weight=0,
        noise_scale=1.0,
        seed=0,
        no_scale=False,
):
    assert n_clients % n_clusters == 0, "n_clients must be divisible by n_clusters"

    n_ds = n_clients // n_clusters
    cluster_labels = []

    X_train = np.zeros((n_clients, n_samples, n_features))
    y_train = np.zeros((n_clients, n_samples, 1))
    X_test = np.zeros((n_clients, n_samples_test, n_features))
    y_test = np.zeros((n_clients, n_samples_test, 1))
    true_weights = np.zeros((n_clusters, n_features))

    rng = np.random.default_rng(seed)

    for i in range(n_clusters):
        w = rng.uniform(-5, 5, size=(n_features, 1))
        true_weights[i] = w.reshape(-1,)

        cluster_X_train = []
        cluster_node_data = []

        # Generate data for all clients in this cluster
        for j in range(n_ds):
            X = rng.normal(0, 1, size=(n_samples + n_samples_test, n_features))
            noise = rng.normal(0, noise_scale, size=(n_samples + n_samples_test, 1))
            noise_w = rng.normal(0, noise_weight, size=(n_features, 1))
            y = X @ (w + noise_w) + noise

            X_t, X_v, y_t, y_v = train_test_split(X, y, train_size=n_samples, test_size=n_samples_test, random_state=seed + i*n_ds + j)
            cluster_X_train.append(X_t)
            cluster_node_data.append((X_t, X_v, y_t, y_v))

        if not no_scale:
            scaler = StandardScaler()
            scaler.fit(np.vstack(cluster_X_train))
            true_weights[i] = w.reshape(-1,) * scaler.scale_

        for idx, (X_t, X_v, y_t, y_v) in enumerate(cluster_node_data):
            X_train[i*n_ds + idx] = scaler.transform(X_t) if not no_scale else X_t
            X_test[i*n_ds + idx] = scaler.transform(X_v) if not no_scale else X_v

            y_train[i*n_ds + idx] = y_t
            y_test[i*n_ds + idx] = y_v

            cluster_labels.append(i)
        
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.as_tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    true_weights = torch.tensor(true_weights, dtype=torch.float32)

    return {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "cluster_labels": cluster_labels,
        "true_weights": true_weights,
        "noise_scale": noise_scale,
    }