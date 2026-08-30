import numpy as np

def impute(X, method="mean"):
    """
    Returns: 2D list with NaN values replaced using the specified method
    """
    X = np.array(X)
    if method == "mean":
        imputed_values = np.nanmean(X, axis=0, keepdims=True)
    elif method == "median":
        imputed_values = np.nanmedian(X, axis=0, keepdims=True)
    imputed_values = np.where(np.isnan(imputed_values), 0, imputed_values)
    return np.where(np.isnan(X), imputed_values, X)