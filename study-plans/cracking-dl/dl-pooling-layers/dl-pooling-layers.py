import numpy as np

def pooling(input: list, pool_size: int, stride: int, pool_type: str) -> list:
    """
    Returns the pooled channel-first tensor.
    """
    input = np.asarray(input, dtype=np.float64)
    channels, h, w = input.shape
    out_channels = channels
    out_h = (h-pool_size)//stride + 1
    out_w = (w-pool_size)//stride + 1
    output = np.empty((out_channels, out_h, out_w))
    for channel in range(out_channels):
        for i in range(out_h):
            for j in range(out_w):
                patch = input[channel, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                if pool_type == "max":
                    output[channel, i, j] = np.max(patch)
                elif pool_type == "average":
                    output[channel, i, j] = np.average(patch)
    return output.round(4).tolist()        