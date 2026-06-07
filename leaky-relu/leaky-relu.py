import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    def func(val, alpha):
        return val if val > 0 else alpha*val

    vectorized_func = np.vectorize(func)

    return vectorized_func(x, alpha)