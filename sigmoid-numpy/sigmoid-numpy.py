import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    def func(val):
        if val > 0:
            return 1 / (1 + np.e**(-val))
        else:
            return np.e**val / (np.e**val +1)
    vectorized_func = np.vectorize(func)

    return vectorized_func(x)