import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    gamma1 = np.asarray(gamma1, dtype=np.float64)
    gamma2 = np.asarray(gamma2, dtype=np.float64)
    beta1 = np.asarray(beta1, dtype=np.float64)
    beta2 = np.asarray(beta2, dtype=np.float64)
    identity = x.copy()
    def bn(x, gamma, beta, epsilon = 1e-5):
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        x_norm = (x - mean) / np.sqrt(var + epsilon)
        return gamma * x_norm + beta
    if mode == "post":
        x1 = np.maximum(0, bn(x @ W1, gamma1, beta1))

        output = np.maximum(0, bn(x1 @ W2, gamma2, beta2) + identity)
    elif mode == "pre":
        x1 = np.maximum(0, bn(x, gamma1, beta1))@W1

        output= np.maximum(0, bn(x1, gamma2, beta2))@W2 + identity
    return {
        "output": output,
        "mode": mode
    }