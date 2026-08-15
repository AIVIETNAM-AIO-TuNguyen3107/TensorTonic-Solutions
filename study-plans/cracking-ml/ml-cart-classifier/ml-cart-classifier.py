import numpy as np
from collections import Counter
def cart_classify(X_train, y_train, X_test, max_depth=5, min_samples=2):
    """
    Returns: list of predicted class labels for each test point
    """
    def gini(arr):
        counts = np.unique(arr, return_counts=True)[1]
        return 1 - np.sum((counts / np.sum(counts)) ** 2)
    def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        feature, current_threshold = None, None
        last_gain = 0
        GS = gini(y)
        if GS == 0:
            return feature, current_threshold
        for j in range(X.shape[-1]):
            feat_copy = np.copy(X[:, j])
            sorted_val = np.sort(np.unique(feat_copy))
            # try every combination of threshold (sorted_val[i] + sorted_val[i+1]) / 2
            for i in range(len(sorted_val) - 1):
                # as hint
                # threshold = (sorted_val[i] + sorted_val[i+1]) / 2

                # as solution
                threshold = sorted_val[i]
                # xj <= threshold go left, xj > threshold go right
                left = y[feat_copy <= threshold]
                right = y[feat_copy > threshold]
                g_sl = gini(left)
                g_sr = gini(right)
                gain = GS - (left.size / y.size)*g_sl - (right.size / y.size)*g_sr
                if gain > last_gain:
                    last_gain = gain
                    feature, current_threshold = j, threshold
        return feature, current_threshold
    def build(X, y, depth):
        if len(y) < min_samples or depth >= max_depth:
            return {"value": Counter(y).most_common()[0][0]}
        feature, threshold = best_split(X, y)
        if feature is None or threshold is None:
            return {"value": Counter(y).most_common()[0][0]}
        left = X[:, feature] <= threshold
        right = X[:, feature] > threshold
        return {
            "feature": feature,
            "threshold": threshold,
            "left": build(X[left], y[left], depth+1),
            "right": build(X[right], y[right], depth+1),
            "value": None
        }
    def predict(tree, test):
        if tree["value"] is not None:
            return tree["value"]
        else:
            if test[tree["feature"]] <= tree["threshold"]:
                return predict(tree["left"], test)
            else:
                return predict(tree["right"], test) 
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    # tree as nested dict with keys feature, threshold, left, right, value
    tree = build(X_train, y_train, 0)
    return [predict(tree, test) for test in X_test]