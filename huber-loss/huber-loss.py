import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    e = np.abs(y_true - y_pred)
    return float(np.mean(np.where(e <= delta, e**2 / 2, delta * (e - delta / 2))))