import numpy as np

def sample_var_std(x):
    """
    Returns: dict with 'variance' and 'std_dev' as floats.
    """
    x = np.array(x)
    variance = np.sum((x - np.mean(x))**2) / (x.shape[-1] - 1)
    std_dev = np.sqrt(variance)
    return {
        "variance": variance,
        "std_dev": std_dev 
    }
