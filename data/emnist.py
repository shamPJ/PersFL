import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
from collections import defaultdict

def load_emnist(root="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.EMNIST(
        root=root,
        split="byclass",   # Includes all 62 classes: 10 digits (0–9) + 52 uppercase and lowercase letters.
        train=True,
        download=True,
        transform=transform
    )

    return train_dataset

def split_emnist_clients(
    dataset,
    num_clients=100,
    s=40,  # percentage of IID data
    seed=42
):
    """
    Split EMNIST into clients using s% IID + (100-s)% label-skewed data.
    """
    assert 0 <= s <= 100
    np.random.seed(seed)

    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients

    # ---- Step 1: shuffle all indices ----
    all_indices = np.random.permutation(num_samples)

    # ---- Step 2: split IID vs skewed pool ----
    iid_size = int(num_samples * s / 100)
    iid_indices = all_indices[:iid_size]
    skew_indices = all_indices[iid_size:]

    # ---- Step 3: distribute IID data evenly ----
    iid_per_client = iid_size // num_clients
    iid_splits = np.array_split(iid_indices[:iid_per_client * num_clients], num_clients)

    # ---- Step 4: sort remaining data by label ----
    labels = np.array([dataset[i][1] for i in skew_indices])
    sorted_indices = skew_indices[np.argsort(labels)]

    skew_per_client = samples_per_client - iid_per_client
    skew_splits = np.array_split(
        sorted_indices[:skew_per_client * num_clients],
        num_clients
    )

    # ---- Step 5: combine IID + skewed data ----
    client_datasets = []
    for i in range(num_clients):
        client_indices = np.concatenate([iid_splits[i], skew_splits[i]])
        client_datasets.append(Subset(dataset, client_indices.tolist()))

    return client_datasets

def get_client_loaders(
    client_datasets,
    batch_size=32,
    shuffle=True
):
    client_loaders = []

    for ds in client_datasets:
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle
        )
        client_loaders.append(loader)

    return client_loaders