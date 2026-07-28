import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if X_train.ndim == 1:
        X_train = X_train[:, None]
        X_test = X_test[:, None]
    dist = np.sqrt(np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :])**2, axis=-1))
    result = np.argsort(dist, axis=-1)[:, :k]
    if result.shape[-1] < k:
        result = np.hstack([result, np.array([-1 for _ in range(k-result.shape[-1])])[None, :]])

    return result