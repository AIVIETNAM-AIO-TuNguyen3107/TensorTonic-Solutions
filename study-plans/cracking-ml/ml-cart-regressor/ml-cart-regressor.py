import numpy as np

def cart_regress(X_train, y_train, X_test, max_depth=5, min_samples=2):
    """
    Returns: list of predicted values rounded to 4 decimal places
    """
    def mse(leaf):
        if len(leaf) == 0:
            return 0
        return np.mean((leaf - np.mean(leaf))**2)
    def best_split(X, y):
        best_feat = None
        best_threshold = None
        best_mse = mse(y)
        for j in range(X.shape[-1]):
            feat = np.copy(X[:, j])
            for t in np.sort(np.unique(feat)):
                left = y[feat <= t]
                right = y[feat > t]
                left_mse = mse(left)
                right_mse = mse(right)
                sl = len(left)
                sr = len(right)
                if sl == 0 or sr == 0:
                    continue
                s = len(y)
                node_mse = (sl / s) * left_mse + (sr / s) * right_mse
                if node_mse < best_mse:
                    best_feat = j
                    best_threshold = t
                    best_mse = node_mse
        return best_feat, best_threshold
    def build_tree(X, y, depth=0):
        if len(y) < min_samples or depth >= max_depth or len(np.unique(y)) == 1:
            return {"leaf": True, "value": float(np.mean(y))}
        best_feat, best_threshold = best_split(X, y)
        if best_feat is None or best_threshold is None:
            return {"leaf": True, "value": float(np.mean(y))}
        left = X[:, best_feat] <= best_threshold
        right = X[:, best_feat] > best_threshold
        return {
            "leaf": False,
            "feat": best_feat,
            "threshold": best_threshold,
            "left": build_tree(X[left], y[left], depth+1),
            "right": build_tree(X[right], y[right], depth+1),
        }

    def inference(tree, test):
        if tree["leaf"]:                              # FIX 4: was `if tree["value"]`
            return tree["value"]
        if test[tree["feat"]] <= tree["threshold"]:
            return inference(tree["left"], test)
        else:
            return inference(tree["right"], test)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    tree = build_tree(X_train, y_train)
    return [round(inference(tree, test), 4) for test in X_test]
