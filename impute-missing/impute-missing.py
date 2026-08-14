import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    def nanmeadian(feat):
        if np.sum(np.isnan(feat)) == len(feat):
            return 0
        else:
            return np.median(feat[~np.isnan(feat)])
    def nanmean(feat):
        if np.sum(np.isnan(feat)) == len(feat):
            return 0
        else:
            return np.mean(feat[~np.isnan(feat)])
    X = np.array(X, dtype=np.float32)
    if strategy == 'mean':
        if X.ndim == 2:
            imputed_data = np.array([nanmean(X[:, c]) for c in range(X.shape[-1])])
        elif X.ndim == 1:
            imputed_data = np.array([nanmean(X)])
    elif strategy == 'median':
        if X.ndim == 2:
            imputed_data = np.array([nanmeadian(X[:, c]) for c in range(X.shape[-1])])
        elif X.ndim == 1:
            imputed_data = np.array([nanmeadian(X)])
            print(imputed_data)
    return np.where(np.isnan(X), imputed_data, X)
