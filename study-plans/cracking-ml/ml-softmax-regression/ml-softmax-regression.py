import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X)
    y = np.array(y)
    n, d = X.shape
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1

    w = np.zeros((d, n_classes))
    b = np.zeros(n_classes)

    for _ in range(n_iters):
        Z = X@w + b
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        P = expZ / np.sum(expZ, axis=1, keepdims=True)

        w -= lr*((1/n)*X.T@(P-Y))
        b -= lr*((1/n)*np.sum(P-Y, axis=0))
    return w, b
