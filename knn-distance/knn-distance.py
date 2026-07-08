import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    if X_train.ndim == 1:
        X_train = X_train[:, np.newaxis]
        X_test = X_test[:, np.newaxis]
    d_xy = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    dist = np.sqrt(np.sum(d_xy**2, axis=-1))

    idx = np.argsort(dist)

    n_train = X_train.shape[0]

    if k <= n_train:
        return idx[:, :k]

    # pad with -1
    result = np.full((X_test.shape[0], k), -1, dtype=int)
    result[:, :n_train] = idx
    return result