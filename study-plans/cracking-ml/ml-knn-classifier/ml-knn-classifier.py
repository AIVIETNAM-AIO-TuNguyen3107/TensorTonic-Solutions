import numpy as np
from collections import Counter, defaultdict
def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    dist = np.sqrt(np.sum((X_test[:, np.newaxis,:] - X_train)**2, axis=-1))
    top_k_indices = np.argsort(dist, axis=-1)[:, :k]
    result = []
    for top_k in top_k_indices:
        counter = Counter(y_train[top_k])
        is_tie = len(set(counter.values())) != len(counter.values())
        if is_tie:
            counter = defaultdict(float)
            for k_index in top_k:
                counter[y_train[k_index]] += 1 / dist[:, k_index][0]
            result.append(sorted(counter)[0])
        else:
            result.append(counter.most_common()[0][0])
    return result