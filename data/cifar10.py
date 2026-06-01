import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

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
    n_clusters, n_clients, n_samples, n_samples_test, n_classes=3, seed=0
):
    """
    CIFAR-10 clustered dataset generator with overlapping classes.
    
    Each client belongs to a cluster. Each cluster is assigned a subset of classes (possibly overlapping).
    Each client gets n_samples train and n_samples_test test examples from its cluster's classes.
    
    Returns:
        dict with keys:
            "train": (X_train, y_train) -- torch tensors of shape (n_clients, n_samples, C,H,W)
            "test":   (X_test, y_test)   -- torch tensors of shape (n_clients, n_samples_test, C,H,W)
            "cluster_labels": list of length n_clients
            "cluster_classes": list of classes per cluster
    """
    rng = np.random.default_rng(seed)
    
    (X_train_full, y_train_full), (X_test_full, y_test_full) = load_cifar10()

    y_train_full = np.array(y_train_full)
    y_test_full  = np.array(y_test_full)

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
    train_indices_by_class = {
        c: np.where(y_train_full == c)[0].tolist() for c in range(n_classes_total)
    }
    test_indices_by_class = {
        c: np.where(y_test_full == c)[0].tolist() for c in range(n_classes_total)
    }
   
    for idxs in train_indices_by_class.values():
        rng.shuffle(idxs)
    for idxs in test_indices_by_class.values():
        rng.shuffle(idxs)

    # Allocate train/val per client
    X_train = torch.zeros((n_clients, n_samples, *X.shape[1:]), dtype=torch.float32)
    y_train = torch.zeros((n_clients, n_samples), dtype=torch.long)
    X_val = torch.zeros((n_clients, n_samples_test, *X.shape[1:]), dtype=torch.float32)
    y_val = torch.zeros((n_clients, n_samples_test), dtype=torch.long)

    samples_per_class_train = max(1, n_samples // n_classes) # to sample approximately equal number of examples per class for each client
    samples_per_class_val   = max(1, n_samples_test // n_classes)

    for i, cluster_id in enumerate(cluster_labels):
        # for ech client, get its cluster's classes and sample from them
        classes = cluster_classes[cluster_id]

         # ---- TRAIN ----
        train_idxs = []
        for c in classes:
            pool = train_indices_by_class[c]
            take = min(len(pool), samples_per_class_train)
            train_idxs += pool[:take]
            train_indices_by_class[c] = pool[take:]

        # fallback if needed
        while len(train_idxs) < n_samples:
            c = rng.choice(classes)
            pool = np.where(y_train_full == c)[0]
            train_idxs.append(rng.choice(pool))

        rng.shuffle(train_idxs)
        train_idxs = train_idxs[:n_samples]

        # ---- VALIDATION ----
        val_idxs = []
        for c in classes:
            pool = test_indices_by_class[c]
            take = min(len(pool), samples_per_class_val)
            val_idxs += pool[:take]
            test_indices_by_class[c] = pool[take:]

        # fallback if needed
        while len(val_idxs) < n_samples_test:
            c = rng.choice(classes)
            pool = np.where(y_test_full == c)[0]
            val_idxs.append(rng.choice(pool))

        rng.shuffle(val_idxs)
        val_idxs = val_idxs[:n_samples_test]

        X_train[i] = X_train_full[train_idxs]
        y_train[i] = torch.tensor(y_train_full[train_idxs])

        X_val[i] = X_test_full[val_idxs]
        y_val[i] = torch.tensor(y_test_full[val_idxs])

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "cluster_labels": cluster_labels.tolist(),
        "cluster_classes": cluster_classes
    }   

def generate_rotated_cifar10(
    n_clusters=4,
    n_clients=100,
    n_samples=500,
    n_samples_test=1000,
    sigma=0.0,
    seed=0,
):
    """
    Generate rotated CIFAR-10 dataset with optional within-cluster angle spread.

    Cluster centers are evenly spaced across [0, 360). With sigma=0 (default)
    every client in a cluster gets exactly the center angle — the hard-cluster
    setup IFCA assumes. With sigma>0 each client's angle is sampled from
    N(cluster_center, sigma^2), blurring cluster boundaries. At sigma~45 deg
    neighbouring clusters start overlapping.

    When sigma=0 the full dataset is rotated once per cluster (fast path).
    When sigma>0 each client's subset is rotated individually (per-client path).

    Setting n_clusters=n_clients and sigma=0 gives every client a unique angle,
    producing continuous heterogeneity with no cluster structure.

    Returns
    -------
    dict with keys:
        "train":             (X_train, y_train)  -- (n_clients, n_samples, C, H, W)
        "test":              (X_test,  y_test)   -- (n_clients, n_samples_test, C, H, W)
        "cluster_labels":    list[int], length n_clients
        "cluster_rotations": list[float], cluster center angles (length n_clusters)
        "client_angles":     list[float], actual per-client rotation angles
    """
    assert n_clients % n_clusters == 0, "n_clients must be divisible by n_clusters"

    rng = np.random.default_rng(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    base_train = datasets.CIFAR10(root="./data_cifar10", train=True,  download=True, transform=transform)
    base_test  = datasets.CIFAR10(root="./data_cifar10", train=False, download=True, transform=transform)

    X_base_train = torch.stack([t[0] for t in base_train])
    y_base_train = torch.tensor([t[1] for t in base_train])
    X_base_test  = torch.stack([t[0] for t in base_test])
    y_base_test  = torch.tensor([t[1] for t in base_test])

    n_sector = 360 // n_clusters
    cluster_centers = [i * n_sector for i in range(n_clusters)]

    n_clients_per_cluster     = n_clients // n_clusters
    n_samples_per_client      = min(X_base_train.shape[0] // n_clients_per_cluster, n_samples)
    n_samples_test_per_client = min(X_base_test.shape[0]  // n_clients_per_cluster, n_samples_test)

    X_train = torch.zeros((n_clients, n_samples_per_client,      *X_base_train.shape[1:]))
    y_train = torch.zeros((n_clients, n_samples_per_client),      dtype=torch.long)
    X_test  = torch.zeros((n_clients, n_samples_test_per_client, *X_base_test.shape[1:]))
    y_test  = torch.zeros((n_clients, n_samples_test_per_client), dtype=torch.long)

    cluster_labels = np.zeros(n_clients, dtype=int)
    client_angles  = np.zeros(n_clients, dtype=float)

    client_idx = 0

    if sigma == 0.0:
        # Fast path: rotate the full dataset once per cluster, then slice.
        for cluster_id, center in enumerate(cluster_centers):
            rot         = transforms.Lambda(lambda x, a=center: transforms.functional.rotate(x, a))
            X_train_rot = torch.stack([rot(img) for img in X_base_train])
            X_test_rot  = torch.stack([rot(img) for img in X_base_test])

            train_indices = rng.permutation(X_train_rot.shape[0])
            test_indices  = rng.permutation(X_test_rot.shape[0])

            for i in range(n_clients_per_cluster):
                t_start, t_end = i * n_samples_per_client,      (i + 1) * n_samples_per_client
                v_start, v_end = i * n_samples_test_per_client, (i + 1) * n_samples_test_per_client

                X_train[client_idx] = X_train_rot[train_indices[t_start:t_end]]
                y_train[client_idx] = y_base_train[train_indices[t_start:t_end]]
                X_test[client_idx]  = X_test_rot[test_indices[v_start:v_end]]
                y_test[client_idx]  = y_base_test[test_indices[v_start:v_end]]

                cluster_labels[client_idx] = cluster_id
                client_angles[client_idx]  = float(center)
                client_idx += 1
    else:
        # Per-client path: each client draws angle ~ N(center, sigma).
        # Pools are regenerated per cluster (matching the sigma=0 fast path) so
        # n_clients_per_cluster * n_samples_*_per_client never exceeds the base set size.
        for cluster_id, center in enumerate(cluster_centers):
            train_pool = rng.permutation(X_base_train.shape[0])
            test_pool  = rng.permutation(X_base_test.shape[0])

            for local_i in range(n_clients_per_cluster):
                angle = float(center + rng.normal(0.0, sigma))

                t_start = local_i * n_samples_per_client
                t_end   = t_start + n_samples_per_client
                v_start = local_i * n_samples_test_per_client
                v_end   = v_start + n_samples_test_per_client

                t_idxs = train_pool[t_start:t_end]
                v_idxs = test_pool[v_start:v_end]

                rot = transforms.Lambda(lambda x, a=angle: transforms.functional.rotate(x, a))

                X_train[client_idx] = torch.stack([rot(X_base_train[j]) for j in t_idxs])
                y_train[client_idx] = y_base_train[t_idxs]
                X_test[client_idx]  = torch.stack([rot(X_base_test[j])  for j in v_idxs])
                y_test[client_idx]  = y_base_test[v_idxs]

                cluster_labels[client_idx] = cluster_id
                client_angles[client_idx]  = angle
                client_idx += 1

    return {
        "train":             (X_train, y_train),
        "test":              (X_test,  y_test),
        "cluster_labels":    cluster_labels.tolist(),
        "cluster_rotations": cluster_centers,
        "client_angles":     client_angles.tolist(),
    }


def generate_rotated_cifar10_shifted(
    n_clusters=4,
    n_clients=24,
    n_samples=500,
    n_samples_test=1000,
    shift_at=15,
    seed=0,
):
    """
    Rotated CIFAR-10 with a sudden mid-training distribution shift.

    Phase 1 (rounds 0 .. shift_at-1): standard rotated setup, cluster angles
    evenly spaced across [0, 360).

    Phase 2 (rounds shift_at .. R-1): the second half of every cluster's
    clients shift to a new rotation angle midway between their cluster center
    and the next one — a genuinely novel angle outside the original cluster set.
    Both training and test data are replaced to reflect the new distribution.

    Returns the standard data dict plus:
        "train_shifted"  : (X_train_shifted, y_train) — same labels, rotated images
        "test_shifted"   : (X_test_shifted,  y_test)  — test images at new angles
        "shift_at"       : int — round at which to swap training data
        "shifted_clients": list[int] — indices of clients whose data changes
    """
    assert n_clients % n_clusters == 0

    base = generate_rotated_cifar10(
        n_clusters=n_clusters,
        n_clients=n_clients,
        n_samples=n_samples,
        n_samples_test=n_samples_test,
        sigma=0.0,
        seed=seed,
    )

    X_train, y_train = base["train"]
    X_test,  y_test  = base["test"]
    cluster_centers  = base["cluster_rotations"]
    n_per_cluster    = n_clients // n_clusters

    n_sector = 360 // n_clusters  # angular gap between cluster centers

    shifted_clients = []
    shift_angles    = [float(cluster_centers[c] + n_sector / 2) for c in range(n_clusters)]
    X_train_shifted = X_train.clone()
    X_test_shifted  = X_test.clone()

    rng = np.random.default_rng(seed + 1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    base_train = datasets.CIFAR10(root="./data_cifar10", train=True,  download=True, transform=transform)
    base_test  = datasets.CIFAR10(root="./data_cifar10", train=False, download=True, transform=transform)
    X_base_train = torch.stack([t[0] for t in base_train])
    y_base_train = torch.tensor(base_train.targets)
    X_base_test  = torch.stack([t[0] for t in base_test])
    y_base_test  = torch.tensor(base_test.targets)

    n_shifted = n_per_cluster // 2
    n_tr       = X_train.shape[1]
    n_te       = X_test.shape[1]

    y_train_shifted = y_train.clone()
    y_test_shifted  = y_test.clone()

    for c in range(n_clusters):
        new_angle = float(cluster_centers[c] + n_sector / 2)  # midpoint to next cluster
        rot       = transforms.Lambda(lambda x, a=new_angle: transforms.functional.rotate(x, a))
        start     = c * n_per_cluster + n_shifted
        end       = (c + 1) * n_per_cluster

        train_pool = rng.permutation(X_base_train.shape[0])[:n_tr * n_shifted]
        test_pool  = rng.permutation(X_base_test.shape[0])[:n_te * n_shifted]

        for local_i, client_idx in enumerate(range(start, end)):
            t_idxs = train_pool[local_i * n_tr : (local_i + 1) * n_tr]
            v_idxs = test_pool [local_i * n_te : (local_i + 1) * n_te]

            X_train_shifted[client_idx] = torch.stack([rot(X_base_train[j]) for j in t_idxs])
            y_train_shifted[client_idx] = y_base_train[t_idxs]
            X_test_shifted[client_idx]  = torch.stack([rot(X_base_test[j])  for j in v_idxs])
            y_test_shifted[client_idx]  = y_base_test[v_idxs]
            shifted_clients.append(client_idx)

    return {
        **base,
        "train_shifted":   (X_train_shifted, y_train_shifted),
        "test_shifted":    (X_test_shifted,  y_test_shifted),
        "shift_at":        shift_at,
        "shift_angles":    shift_angles,
        "shifted_clients": shifted_clients,
    }
