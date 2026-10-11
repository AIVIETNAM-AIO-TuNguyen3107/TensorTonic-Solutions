import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    ssr = np.sum((y_true - y_pred) ** 2)

    if sst == 0:
        return 1.0 if ssr == 0 else 0.0

    return float(1 - ssr / sst)