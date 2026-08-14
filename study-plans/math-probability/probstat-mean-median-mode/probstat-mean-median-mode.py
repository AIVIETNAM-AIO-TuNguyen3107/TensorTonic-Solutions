import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Returns: dict with 'mean', 'median', 'mode' as floats.
    """
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "mode": Counter(x).most_common()[0][0]
    }