import numpy as np

def vanilla_rnn(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, W_hy: np.ndarray, b_h: np.ndarray,
                b_y: np.ndarray) -> dict:
    """
    Returns outputs and final_hidden_state as float64 arrays.
    """
    h = h_0
    N, H = h.shape
    N, T, D = X.shape
    O, H = W_hy.shape
    outputs = np.zeros((N, T, O))
    for step in range(T):
        input_contribution = X[:, step, :]@W_xh.T
        hidden_contribution = h@W_hh.T
        h = np.tanh(input_contribution+hidden_contribution+b_h)
        outputs[:, step, :] = h@W_hy.T + b_y
    return {
        "outputs": outputs,
        "final_hidden_state": h
    }