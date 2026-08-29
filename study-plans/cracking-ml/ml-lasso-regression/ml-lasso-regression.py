import numpy as np
def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    X = np.array(X)
    y = np.array(y)
    n, d = X.shape
    w = np.zeros(d)
    b = 0

    for _ in range(epochs):
        y_hat = X@w + b
        error = y_hat - y
        w -= lr*((2/n)*X.T@(error) + alpha*np.sign(w))
        b -= lr*((2/n)*np.sum(error))

    return w, b