import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    conv1 = np.asarray(conv1, dtype=np.float64)
    W1_b1 = np.asarray(W1_b1, dtype=np.float64)
    W2_b1 = np.asarray(W2_b1, dtype=np.float64)
    W1_b2 = np.asarray(W1_b2, dtype=np.float64)
    W2_b2 = np.asarray(W2_b2, dtype=np.float64)
    Ws_b2 = np.asarray(Ws_b2, dtype=np.float64)

    x = np.maximum(0, x @ conv1)

    blocks = [
        (W1_b1, W2_b1, None),
        (W1_b2, W2_b2, Ws_b2)
    ]
    for W1, W2, Ws in blocks:
        identity = x.copy()
        if Ws is not None:
            identity = x@Ws
        x = np.maximum(0, np.maximum(0, x@W1)@W2 + identity)
    return (x@fc).round(4)
