import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    k = num_classes
    y = np.array(y)
    y = np.eye(k)[y]

    def stable_softmax(z):
        max = np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z-max)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    X = np.array(X)
    n, d = X.shape
    w = np.zeros((d, k))
    b = np.zeros(k)
    for _ in range(n_iters):
        z = X@w + b
        y_hat = stable_softmax(z)
        gradient = y_hat-y
        w -= lr*(1/n)*X.T@gradient
        b -= lr*np.mean(gradient.T, axis=-1)
    return w, b