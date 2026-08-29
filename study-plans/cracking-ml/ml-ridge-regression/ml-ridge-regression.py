import numpy as np
def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X = np.array(X)
    y = np.array(y)
    n, d = X.shape
    w = np.zeros(d)
    b = 0

    for _ in range(epochs):
        y_hat = X@w + b
        g = y_hat - y # always go reverse gradient
        w -= lr*((2/n)*X.T@(g) + 2*alpha*w)
        b -= lr*(2/n)*np.sum(g)
    return w, b
