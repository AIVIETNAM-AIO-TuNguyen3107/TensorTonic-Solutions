import numpy as np

def log_loss(y_true, y_pred):
    """
    Returns: float
    """
    n = len(y_true)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return np.sum([(-1/n)*(y_i*np.log(pi) + (1-y_i)*np.log(1-pi)) for y_i, pi in zip(y_true, y_pred, strict=True)]) or 0
