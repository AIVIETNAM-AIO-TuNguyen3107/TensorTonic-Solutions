import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    arr = np.array(data, dtype=np.float64)
    return np.vstack([
        arr > threshold,
        np.where(np.expand_dims(np.any(arr > threshold, axis=1), axis=1), arr, 0),
        np.where(np.expand_dims(np.all(arr > threshold, axis=1), axis=1), arr, 0)
    ]).reshape((3, arr.shape[0], arr.shape[1]))