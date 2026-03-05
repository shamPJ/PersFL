import numpy as np
from sklearn.model_selection import train_test_split

def generate_data(n_clusters, n_clients, n_samples, n_features, noise_scale=1.0):
    
    """
    
    Function to create a noisy Gaussian regression datasets multivar Gaussian ~N(0,I). 
    Datasets within the cluster share the same true weight vector. 
    For each node we create training (size = n_samples) and validation (size = 100) ds.
    
    Args:
    : n_clusters  : int, number of clusters
    : n_clients   : int, number of local datasets / clients 
    : n_samples   : number of samples in a local dataset
    : n_features  : number of features of a datapoint
    : noise_scale : scale of normal distribution used to generate data noise
    
    Out:
    : ds_train       : tuple (X_train, y_train), where X array of shape (n_nodes, m_i, d) and y - (n_nodes, m_i, 1)
    : ds_val         : tuple (X_val ,y_val), where X array of shape (n_nodes, m_i, d) and y - (n_nodes, m_i, 1)
    : cluster_labels : list of len(n_clients) cluster assignments for each local dataset 
    : true_weights   : array of shape (n_clusters, n_features), true weight vector for each cluster

    """

    assert n_clients % n_clusters == 0, \
    "n_clients must be divisible by n_clusters"

    # equal n.o. clients per cluster
    n_ds = int(n_clients / n_clusters)
    # Lists to store and return outputs
    cluster_labels = []
    m_val = 100 # hardcoded size of the validation set
    X_train, y_train = np.zeros((n_clients, n_samples, n_features)), np.zeros((n_clients, n_samples, 1))
    X_val, y_val = np.zeros((n_clients, m_val, n_features)), np.zeros((n_clients, m_val, 1))
    
    true_weights   = np.zeros((n_clusters, n_features))

    node_id = 0
    for i in range(n_clusters):
    
        # Sample true weight vector for cluster i
        #w = np.random.normal(0, 1, size=(n_features,1))
        w = np.random.uniform(-5, 5, size=(n_features,1))
        true_weights[i] = w.reshape(-1,)

        for j in range(n_ds):
            # Sample datapoints from multivar Gaussian ~N(0,I)
            X = np.random.normal(0, 1.0, size=(n_samples + m_val, n_features))
            
            # Sample noise 
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=(n_samples + m_val, 1))
            
            # Noisy Gaussian regression
            y = X@w + noise

            # Split train vs val
            X_t, X_v, y_t, y_v = train_test_split(X, y, train_size=n_samples, test_size=m_val)
            X_train[node_id] = X_t
            y_train[node_id] = y_t
            
            X_val[node_id] = X_v
            y_val[node_id] = y_v
            
            cluster_labels.append(i)
            node_id += 1

    return {
            "train": (X_train, y_train),
            "cluster_labels": cluster_labels,
            "true_weights": true_weights
            }
