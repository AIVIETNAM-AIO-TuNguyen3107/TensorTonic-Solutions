import numpy as np
from collections import Counter
def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    l2_from_test_to_each_train = np.sqrt(np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :])**2, axis=-1))

    sorted_idx_for_each_test = np.argsort(l2_from_test_to_each_train, axis=-1)

    result = []
    for sorted_idx in sorted_idx_for_each_test:
        candidate_labels = y_train[sorted_idx[:k]]
        unique_labels = set(candidate_labels)

        counter = Counter(candidate_labels)
        most_common = counter.most_common()
        is_tie = len([common for common in most_common if common[1] == most_common[0][1]]) > 1
        if is_tie:
            result.append(min(unique_labels))
        else:
            result.append(most_common[0][0])
    return result