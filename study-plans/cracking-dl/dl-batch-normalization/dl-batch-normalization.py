import numpy as np

def batch_norm(X: list, gamma: list, beta: list, running_mean: list, running_var: list, mode: str) -> dict:
    """
    Returns normalized output and resulting running statistics.
    """
    X = np.array(X, dtype=np.float64)
    gamma = np.array(gamma, dtype=np.float64)
    beta = np.array(beta, dtype=np.float64)
    running_mean = np.array(running_mean, dtype=np.float64)
    running_var = np.array(running_var, dtype=np.float64)
    epsilon = 10**-5
    if mode == "test":
        return {
            "output": gamma*((X - running_mean) / np.sqrt(running_var + epsilon)) + beta,
            "running_mean": running_mean,
            "running_var": running_var
        }
    elif mode == "train":
        mean = np.mean(X, axis=0)
        var = np.var(X, axis=0, ddof=0)
        running_mean = 0.9*running_mean + 0.1*mean
        running_var = 0.9*running_var + 0.1*var
        return {
            "output": gamma*((X - mean) / np.sqrt(var + epsilon)) + beta,
            "running_mean": running_mean,
            "running_var": running_var
        }