import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos = np.arange(seq_length).reshape(seq_length, -1)
    i = np.arange(d_model).reshape(-1, d_model)
    pair_index = i // 2
    divider = 10000 ** (2*pair_index/d_model)
    return np.where(
        i % 2 == 0,
        np.sin(pos/divider),
        np.cos(pos/divider)
    )
