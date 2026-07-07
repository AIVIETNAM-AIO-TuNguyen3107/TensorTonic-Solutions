import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pair_index = np.arange(0, d_model)
    position = np.arange(0, seq_length)[:, None]
    return np.where(
        pair_index % 2 == 0,
        np.sin(position / 10000 ** (2*(pair_index//2)/d_model)),
        np.cos(position / 10000 ** (2*(pair_index//2)/d_model))
    )
