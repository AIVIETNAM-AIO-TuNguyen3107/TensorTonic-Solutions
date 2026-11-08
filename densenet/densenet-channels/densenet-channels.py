import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int,
                            block_layers: list, compression: float) -> torch.Tensor:
    """
    Returns the int64 channel count at every DenseNet stage.
    """
    results = [stem_channels]

    for layer in block_layers:
        results.append(math.floor(results[-1] + growth_rate*layer))
        results.append(math.floor(results[-1]*compression))

    return torch.tensor(results[:-1], dtype=torch.int64)