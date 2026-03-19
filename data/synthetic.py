import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_data(n_clusters, n_clients, n_samples, n_samples_val, n_features, noise_weight=0, noise_scale=1.0, seed=0):
    assert n_clients % n_clusters == 0, "n_clients must be divisible by n_clusters"

    n_ds = n_clients // n_clusters
    cluster_labels = []

    X_train = np.zeros((n_clients, n_samples, n_features))
    y_train = np.zeros((n_clients, n_samples, 1))
    X_val = np.zeros((n_clients, n_samples_val, n_features))
    y_val = np.zeros((n_clients, n_samples_val, 1))
    true_weights = np.zeros((n_clusters, n_features))

    rng = np.random.default_rng(seed)

    for i in range(n_clusters):
        w = rng.uniform(-5, 5, size=(n_features, 1))
        true_weights[i] = w.reshape(-1,)

        cluster_X_train = []
        cluster_node_data = []

        # Generate data for all clients in this cluster
        for j in range(n_ds):
            X = rng.normal(0, 1, size=(n_samples + n_samples_val, n_features))
            noise = rng.normal(0, noise_scale, size=(n_samples + n_samples_val, 1))
            noise_w = rng.normal(0, noise_weight, size=(n_features, 1))
            y = X @ (w + noise_w) + noise

            X_t, X_v, y_t, y_v = train_test_split(X, y, train_size=n_samples, test_size=n_samples_val, random_state=seed + i*n_ds + j)
            cluster_X_train.append(X_t)
            cluster_node_data.append((X_t, X_v, y_t, y_v))

        # Fit scaler on all cluster training data
        scaler = StandardScaler()
        scaler.fit(np.vstack(cluster_X_train))

        # Transform each client
        for idx, (X_t, X_v, y_t, y_v) in enumerate(cluster_node_data):
            X_train[i*n_ds + idx] = scaler.transform(X_t)
            X_val[i*n_ds + idx] = scaler.transform(X_v)
            y_train[i*n_ds + idx] = y_t
            y_val[i*n_ds + idx] = y_v

            cluster_labels.append(i)

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "cluster_labels": cluster_labels,
        "true_weights": true_weights
    }