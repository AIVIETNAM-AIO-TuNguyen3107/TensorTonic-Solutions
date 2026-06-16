import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch, n, d_model = Q.shape
    d_k = d_model // num_heads
    head_Q = (Q@W_q).reshape(batch, n, num_heads, d_k).transpose((0, 2, 1, 3))
    head_K = (K@W_k).reshape(batch, n, num_heads, d_k).transpose((0, 2, 1, 3))
    head_V = (V@W_v).reshape(batch, n, num_heads, d_k).transpose((0, 2, 1, 3))
    return (softmax(head_Q@head_K.swapaxes(-1, -2)/np.sqrt(d_k))@head_V).transpose(0, 2, 1, 3).reshape(batch, n, d_model)@W_o
