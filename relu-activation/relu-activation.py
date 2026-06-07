import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    return np.maximum(x, 0)