import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    return [value if value > 0 else alpha*(np.e**value-1) for value in x]