import numpy as np

def distance_metric(x, y, metric, p=2):
    """
    Compute the distance between vectors x and y using the specified metric.
    Returns: float rounded to 4 decimal places
    """
    x = np.array(x)
    y = np.array(y)
    if metric == "euclidean":
        return np.sqrt(np.sum((x-y)**2))
    elif metric == "manhattan":
        return np.sum(abs(x-y))
    elif metric == "cosine":
        cosine_dist = 1 - (x@y / (np.linalg.norm(x)*np.linalg.norm(y)))
        return cosine_dist if not np.isnan(cosine_dist) else 0.0
    elif metric == "chebyshev":
        return np.max(abs(x-y))
    elif metric == "minkowski":
        return (np.sum(abs(x-y)**p))**(1/p)