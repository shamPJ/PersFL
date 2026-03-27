import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

def load_cifar10(root="./data_cifar10"):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) 
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor(train_dataset.targets)
    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor(test_dataset.targets)

    X = torch.cat([X_train, X_test], dim=0)
    y = torch.cat([y_train, y_test], dim=0)
    return X, y

def generate_clustered_cifar10(
    n_clusters, n_clients, n_samples, n_samples_val, n_classes=3, seed=0
):
    """
    CIFAR-10 clustered dataset generator with overlapping classes.
    
    Each client belongs to a cluster. Each cluster is assigned a subset of classes (possibly overlapping).
    Each client gets n_samples train and n_samples_val validation examples from its cluster's classes.
    
    Returns:
        dict with keys:
            "train": (X_train, y_train) -- torch tensors of shape (n_clients, n_samples, C,H,W)
            "val":   (X_val, y_val)     -- torch tensors of shape (n_clients, n_samples_val, C,H,W)
            "cluster_labels": list of length n_clients
            "cluster_classes": list of classes per cluster
    """
    rng = np.random.default_rng(seed)
    X, y = load_cifar10()
    y = np.array(y)
    n_classes_total = 10

    # Assign classes to clusters (allow overlap)
    cluster_classes = [] 
    # e.g. [[4, 0, 5, 6], [4, 1, 0, 5], [5, 4, 6, 1]]
    # for n_clusters=3, n_classes=4, n_clients=10
    for _ in range(n_clusters):
        classes = rng.choice(n_classes_total, size=n_classes, replace=False)
        print("n_clusters,n_classes, classes", n_clusters,n_classes, classes)
        cluster_classes.append(classes.tolist())

    # Assign clients to clusters
    clients_per_cluster = n_clients // n_clusters
    cluster_labels = [] # e.g. [2 0 1 0 1 1 2 0 1 2] for n_clusters=3, n_clients=10
    for cluster_id in range(n_clusters):
        cluster_labels += [cluster_id] * clients_per_cluster # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # add remaining clients to random clusters if n_clients not divisible by n_clusters
    # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, 1] to ensure approx. equal distribution of clients across clusters
    cluster_labels += rng.choice(n_clusters, n_clients - len(cluster_labels)).tolist() 
    cluster_labels = np.array(cluster_labels)
    rng.shuffle(cluster_labels)

    # Prepare per-class indices
    indices_by_class = {c: np.where(y == c)[0].tolist() for c in range(n_classes_total)}
    for idxs in indices_by_class.values():
        rng.shuffle(idxs)

    # Allocate train/val per client
    X_train = torch.zeros((n_clients, n_samples, *X.shape[1:]), dtype=torch.float32)
    y_train = torch.zeros((n_clients, n_samples), dtype=torch.long)
    X_val = torch.zeros((n_clients, n_samples_val, *X.shape[1:]), dtype=torch.float32)
    y_val = torch.zeros((n_clients, n_samples_val), dtype=torch.long)

    samples_per_class = (n_samples + n_samples_val) // n_classes # to sample approximately equal number of examples per class for each client
    for i, cluster_id in enumerate(cluster_labels):
        # for ech client, get its cluster's classes and sample from them
        classes = cluster_classes[cluster_id]
        client_indices = []

        # Collect samples from cluster's classes
        for c in classes:
            idxs_c = indices_by_class[c]
            take = min(len(idxs_c[:samples_per_class]), n_samples + n_samples_val - len(client_indices))
            client_indices += idxs_c[:take]
            indices_by_class[c] = idxs_c[take:] # remove taken indices

        rng.shuffle(client_indices)
        # If not enough samples, sample with replacement from cluster's classes
        while len(client_indices) < n_samples + n_samples_val:
            c = rng.choice(classes)
            idxs_c = np.where(y == c)[0].tolist()
            client_indices.append(rng.choice(idxs_c))

        train_idxs = client_indices[:n_samples]
        val_idxs = client_indices[n_samples:n_samples + n_samples_val]

        X_train[i] = X[train_idxs]
        y_train[i] = torch.tensor(y[train_idxs])
        X_val[i] = X[val_idxs]
        y_val[i] = torch.tensor(y[val_idxs])

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "cluster_labels": cluster_labels.tolist(),
        "cluster_classes": cluster_classes
    }

# ----------------------
# Example usage
# ----------------------
dataset = generate_clustered_cifar10(
    n_clusters=5,
    n_clients=50,
    n_samples=50,
    n_samples_val=20,
    n_classes=3,
    seed=42
)

print("Train shape:", dataset["train"][0].shape)
print("Val shape:", dataset["val"][0].shape)
print("Cluster classes:", dataset["cluster_classes"])
print("Cluster labels per client:", dataset["cluster_labels"][:10])
