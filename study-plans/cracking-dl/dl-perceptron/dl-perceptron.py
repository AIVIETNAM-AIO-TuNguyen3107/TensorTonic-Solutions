import numpy as np


def perceptron(X: list, y: list, lr: float = 0.1, epochs: int = 100) -> tuple:
    """
    Returns the trained weight list and bias.
    """
    X = np.array(X, dtype=np.float64)
    n, d = X.shape
    y = np.array(y, dtype=np.float64)
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        for x, target in zip(X, y):
            logits = x @ w + b
            y_hat = 1 if logits >= 0 else 0
            gradient = y_hat - target
            w -= lr * gradient * x
            b -= lr * gradient
    return w, b