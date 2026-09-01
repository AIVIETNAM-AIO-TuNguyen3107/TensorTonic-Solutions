import numpy as np

def kfold_cv(X, y, model_fn, k=5, seed=42):
    """
    Returns: tuple of (list of per-fold accuracies, mean accuracy)
    """
    rng = np.random.RandomState(seed=seed)
    X = np.array(X)
    y = np.array(y)
    n, d = X.shape
    indices = rng.permutation(n)

    kfold = np.array_split(indices, k)
    result = []
    for i in range(k):
        X_val = X[kfold[i]]
        y_val = y[kfold[i]]
        X_train = np.concatenate([X[kfold[j]] for j in range(k) if j!=i ])
        y_train = np.concatenate([y[kfold[j]] for j in range(k) if j!=i ])
        pred_func = model_fn(X_train, y_train)
        preds = pred_func(X_val)
        result.append(round(np.mean(preds == y_val), 4))
    return result, round(np.mean(result), 4)