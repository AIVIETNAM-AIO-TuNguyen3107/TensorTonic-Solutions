import numpy as np

def row_summary(data, threshold:int):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    arr = np.array(data)

    mask = arr > threshold
    return np.array(
        [
            mask,
            np.where(mask.any(axis=1, keepdims=True), arr, 0),
            np.where(mask.all(axis=1, keepdims=True), arr, 0)
        ]
    )
