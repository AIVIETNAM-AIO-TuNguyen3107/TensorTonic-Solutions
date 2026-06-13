import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    return nn.Embedding(vocab_size, d_model) # vocab_size rows/tensors with size of columns/d_model

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    embedding_lookedup = embedding(tokens) # tokens should be in ints, nn.Embedding expecet a LongTensor
    return embedding_lookedup * math.sqrt(d_model)