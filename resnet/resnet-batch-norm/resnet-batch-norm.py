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
    def bn(x, gamma, beta, eps = 1e-5):
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    if mode == "post":
        output = np.maximum(0, bn(np.maximum(0, bn(x @ W1, gamma1, beta1)) @ W2, gamma2, beta2) + x)
    elif mode == "pre":
        output= np.maximum(0, bn(np.maximum(0, bn(x, gamma1, beta1))@W1, gamma2, beta2))@W2 + x
    return {
        "output": output,
        "mode": mode
    }