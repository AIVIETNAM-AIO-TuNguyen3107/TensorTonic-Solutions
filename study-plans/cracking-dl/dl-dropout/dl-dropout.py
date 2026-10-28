import numpy as np

def dropout(X: list, mask: list, drop_prob: float, mode: str) -> list:
    """
    Returns the train-mode dropout result or unchanged test input.
    """
    X = np.array(X, dtype=np.float64)
    if mode =="test":
        return X
    elif mode == "train":
        return X*mask / (1-drop_prob)