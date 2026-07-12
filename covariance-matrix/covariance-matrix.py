import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # each column in this problem is a feature
    X = np.array(X)
    if len(X.shape) != 2 or X.shape[0] < 2:
        return None
    # we want to calculate observered value with each feature (column) mean
    X_centered = X - np.mean(X, axis = 0)

    # use matmul, not hamadard
    # sum((feature1_i - feature1_mean)*(feature2_i - feature2_mean))
    cov = X_centered.T @ X_centered

    return cov / (X.shape[0] - 1)