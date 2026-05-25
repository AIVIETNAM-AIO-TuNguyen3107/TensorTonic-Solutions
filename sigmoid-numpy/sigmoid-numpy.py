import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    def formula(x):
        if x > 0:
            return 1 / (1 +  np.e**(-x))
        else:
            return np.e**x / (1 + np.e**x)

    vectorize = np.vectorize(formula)
    return vectorize(x)