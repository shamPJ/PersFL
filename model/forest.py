import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForest:
    """
    Wrapper to unify sklearn model with torch-based pipeline
    """

    def __init__(self, max_depth=5, n_estimators=100, seed=0):
        self.model = RandomForestRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=seed
        )
        
    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x

    def fit(self, X, y, sample_weight=None):
        X = self._to_numpy(X)
        y = self._to_numpy(y).reshape(-1)
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        X = self._to_numpy(X)
        pred = self.model.predict(X)
        return pred