import numpy as np

def feature_scale(X, method="minmax"):
    """
    Returns: 2D list of scaled values
    """
    X = np.array(X)
    if method == "minmax":
        min = np.min(X, axis=0, keepdims=True)
        max = np.max(X, axis=0, keepdims=True)
        d = max - min
        d = np.where(d==0, 1, d)
        return (X - min) / d
    elif method == "standard":
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        std = np.where(std==0, 1, std)
        return (X - mean) / std