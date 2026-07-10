import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    d = np.arange(0, d_model)[None, :]
    p = np.arange(0, seq_len)[:, None]

    pair_index = d // 2
    return np.where(
        d % 2 == 0,
        np.sin(p / (base ** (2*pair_index/d_model))),
        np.cos(p / (base ** (2*pair_index/d_model))),
    )