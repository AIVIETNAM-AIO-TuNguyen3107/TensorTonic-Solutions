import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.array(x)
    n = x.shape[0]
    x_mean = np.mean(x)
    s = np.sqrt((1/(n-1)) * np.sum((x-x_mean)**2))

    t = (x_mean-mu0) / (s/np.sqrt(n))

    return t