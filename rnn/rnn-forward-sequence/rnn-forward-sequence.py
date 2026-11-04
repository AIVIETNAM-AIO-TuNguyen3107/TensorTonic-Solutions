import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, b_h: np.ndarray) -> dict:
    """
    Returns hidden_states and final_hidden_state as float64 arrays.
    """
    h = h_0.copy()
    N, H = h.shape
    N, T, D = X.shape
    hidden_states = np.zeros((N, T, H))
    for step in range(T):
        input_contribution = X[:, step, :]@W_xh.T # (N, D) @ (H, D).T
        recurrent_contribution = h@W_hh.T
    
        h = np.tanh(input_contribution+recurrent_contribution+b_h)
        hidden_states[:, step, :] = h
    return {
        "hidden_states": hidden_states,
        "final_hidden_state": h
    }