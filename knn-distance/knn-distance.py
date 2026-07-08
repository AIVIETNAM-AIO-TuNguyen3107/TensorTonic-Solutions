import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise Euclidean distances between test samples and training samples,
    then return the indices of the k nearest training samples for each test sample.
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # If the data is 1D, treat each value as a sample with one feature.
    #
    # Example:
    #   X_train: (5,) -> (5, 1)
    #   X_test : (3,) -> (3, 1)
    if X_train.ndim == 1:
        X_train = X_train[:, np.newaxis]
        X_test = X_test[:, np.newaxis]

    # Add singleton dimensions to enable broadcasting.
    #
    # X_test : (n_test, 1, n_features)
    # X_train: (1, n_train, n_features)
    #
    # Broadcasting expands them to:
    #
    #   (n_test, n_train, n_features)
    #
    # Each element d_xy[i, j] is the feature-wise difference between
    # test sample i and training sample j.
    d_xy = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]

    # Euclidean distance:
    #   1. Square each feature difference.
    #   2. Sum over the feature dimension.
    #   3. Take the square root.
    #
    # Result shape: (n_test, n_train)
    dist = np.sqrt(np.sum(d_xy**2, axis=-1))

    # Sort training indices for each test sample by increasing distance.
    #
    # Result shape: (n_test, n_train)
    idx = np.argsort(dist)

    n_train = X_train.shape[0]

    if k <= n_train:
        return idx[:, :k]

    # If k is larger than the number of training samples,
    # pad the remaining positions with -1.
    result = np.full((X_test.shape[0], k), -1, dtype=int)
    result[:, :n_train] = idx
    return result