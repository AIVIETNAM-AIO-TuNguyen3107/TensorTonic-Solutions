import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, W_xh: np.ndarray,
             W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Returns the float64 next hidden state.
    """
    input_contribution = x_t@W_xh.T
    recurrent_contribution = h_prev@W_hh.T
    return np.tanh(input_contribution + recurrent_contribution + b_h)