from math import inf
import numpy as np
from collections import Counter, defaultdict
def bagging_classify(X_train, y_train, X_test, n_estimators=10, max_depth=5, seed=42):
    """
    Returns: list of predicted class labels for each test point
    """
    def gini(y):
        _, counts = np.unique_counts(y)
        p = counts / len(y)
        return 1 - np.sum(p**2)
    def best_split(X, y):
        GS = gini(y)
        best_feature = None
        best_threshold = None
        best_gain = -inf
        for j in range(X.shape[-1]):
            feat = np.copy(X[:, j])
            for t in np.sort(np.unique(feat)):
                left = feat[feat <= t]
                right = feat[feat > t]
                gini_left = gini(y[feat<=t])
                gini_right = gini(y[feat>t])
                sl = len(left)
                sr = len(right)
                s = len(y)
                gain = GS - (sl/s)*gini_left - (sr/s)*gini_right
                if gain > 0 and gain > best_gain:
                    best_feature = j
                    best_threshold = t
                    best_gain = gain
        return best_feature, best_threshold
    def build_tree(X, y, depth):
        if len(y) <= 2 or depth > max_depth or gini(y) == 0:
            return {"leaf": True, "value": Counter(y).most_common()[0][0]}
        feat, threshold = best_split(X, y)
        if feat is None or threshold is None:
            return {"leaf": True, "value": Counter(y).most_common()[0][0]}
        left_mask = X[:, feat]<=threshold
        right_mask = X[:, feat]>threshold
        return {
            "leaf": False,
            "feat": feat,
            "threshold": threshold,
            "left": build_tree(X[left_mask], y[left_mask], depth+1),
            "right": build_tree(X[right_mask], y[right_mask], depth+1),
            "value": None
        }
    def build_estimators(X, y):
        n, d = X.shape
        rng = np.random.RandomState(seed)
        estimators = []
        for _ in range(n_estimators):
            bootstrap_indices = rng.choice(n, n, replace=True)
            estimators.append(build_tree(X[bootstrap_indices], y[bootstrap_indices], 0))
        return estimators
    def predict(estimator, X_test):
        if estimator["leaf"]:
            return estimator["value"]
        else:
            if X_test[estimator["feat"]] <= estimator["threshold"]:
                return predict(estimator["left"], X_test)
            else:
                return predict(estimator["right"], X_test)
    def voting(estimators, X_test):
        vote_count = defaultdict(int)
        for estimator in estimators:
            pred_class = predict(estimator, X_test)
            vote_count[pred_class] += 1
        result = Counter(vote_count)
        return result.most_common()[0][0]
    X = np.array(X_train)
    y = np.array(y_train)
    X_test = np.array(X_test)
    estimators = build_estimators(X, y)
    return [voting(estimators, test) for test in X_test]
