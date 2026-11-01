import numpy as np

def loss_functions(y_true: list, y_pred: list, loss_type: str) -> float:
    """
    Returns the selected mean loss.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    if loss_type == "mse":
        return np.mean((y_pred-y_true)**2)
    elif loss_type == "bce":
        epsilon = 10**-15
        y_pred = np.where(y_pred==0, epsilon, np.where(y_pred==1, y_pred-epsilon, y_pred))
        return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    elif loss_type == "cce":
        max_logits = np.max(y_pred, axis=1, keepdims=True)
        log_probs = (y_pred - max_logits) - np.log(
            np.sum(np.exp(y_pred - max_logits), axis=1, keepdims=True)
        )
        return -np.mean(log_probs[np.arange(len(y_true)), y_true.astype(int)])
    elif loss_type == "hinge":
        return np.mean(np.maximum(0, 1 - y_pred*y_true))