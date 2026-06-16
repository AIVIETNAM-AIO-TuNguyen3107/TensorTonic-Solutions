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
    if Q.ndim == 2: # batch support
        Q = Q[np.newaxis, ...]
        K = K[np.newaxis, ...]
        V = V[np.newaxis, ...]
    
    batch, seq, d_model = Q.shape
    
    # each head will get d_k dimensions, 
    # conceptually, Wq is (d_model, d_k), 
    # but for computation convenience, Wq is (d_model, d_model) then split into n_heads later by split_head
    d_k = d_model // num_heads 
    
    def split_head(matrix:np.ndarray):
        return matrix.reshape(
            batch, seq, num_heads, d_k # b n h d_k
        ).transpose(
            (0, 2, 1, 3) # b h n d_k
        ) 
    
    head_Q = split_head(Q@W_q)
    head_K = split_head(K@W_k)
    head_V = split_head(V@W_v) 
    
    # use swapaxes to transpose K to b h d_k n
    scores = head_Q@head_K.swapaxes(-1, -2) / np.sqrt(d_k) # b h n n
    attention = softmax(scores)@head_V # b h n n @ b h n d_k = b h n d_k
    
    # tranpose to b n h d_k then reshape to b n d_model
    head_concat = attention.transpose((0, 2, 1, 3)).reshape(batch, seq, d_model) 
    return head_concat@W_o