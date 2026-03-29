from sklearn.tree import DecisionTreeRegressor
import numpy as np

class DecisionTree:
    """
    Decision Tree with callable interface: model(X) -> predictions
    """
    def __init__(self, max_depth=None, seed=42):
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=seed
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def __call__(self, X):
        """
        Makes the instance callable: tree(X)
        """
        return self.model.predict(X)