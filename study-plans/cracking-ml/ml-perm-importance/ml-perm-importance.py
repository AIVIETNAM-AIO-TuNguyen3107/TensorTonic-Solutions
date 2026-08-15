import numpy as np

def permutation_importance(X, y, predict_fn, n_repeats=5, seed=42):
    """
    Returns: list of importance scores (one per feature) rounded to 4 decimal places
    """
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    rng = np.random.RandomState(seed)
    baseline = np.mean(predict_fn(X) == y)

    result = []
    for j in range(X.shape[-1]):
        accuracy_drop = []
        for _ in range(n_repeats):
            X_cp = np.copy(X)
            X_cp[:, j] = rng.permutation(X_cp[:, j]) # permutate on fresh copy of column j
            accuracy_drop.append(baseline - np.mean(predict_fn(X_cp) == y))
        result.append(round(np.mean(accuracy_drop), 4))
    return result