import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    def sigmoid(z):
        if z > 0:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (1 + np.exp(z))
    vectorize = np.vectorize(sigmoid)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    for _ in range(n_iters):
        y_hat = vectorize(X@w + b)
        w -= lr*(1/n)*X.T@(y_hat - y)
        b -= lr*(1/n)*np.sum(y_hat - y)
    return w, b