import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    @np.vectorize
    def func(val):
        return 1 / (1 + np.e**(-val)) if val > 0 else np.e**(val) / (1 + np.e**(val))

    return func(x)