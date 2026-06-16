import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    # with this problem, the d_k equal Q last dimension.
    # In multi-head attention, Q last dimension usually equal d_model then split into heads d_model = d_k * heads
    batch, seq_length, d_k = Q.shape
    # scaled dot-product, feature correlations, not token scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # softmax row-wise, over the key dimension of the last axis
    attention = torch.matmul(F.softmax(scores, dim=-1), V)
    return attention