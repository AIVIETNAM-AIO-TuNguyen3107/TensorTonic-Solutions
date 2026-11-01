import numpy as np


def conv2d(
    input: list, filters: list, bias=None, padding: int = 0, stride: int = 1
) -> list:
    """
    Returns the channel-first cross-correlation output.
    """
    input = np.array(input, dtype=np.float64)
    _, h, w = input.shape
    padded = np.pad(input, ((0, 0), (padding, padding), (padding, padding)))
    filters = np.asarray(filters, dtype=np.float64)
    out_channels, in_channels, k_h, k_w = filters.shape
    bias = (
        np.zeros((out_channels)) if bias is None else np.asarray(bias, dtype=np.float64)
    )
    h_out = (h + 2 * padding - k_h) // stride + 1
    w_out = (w + 2 * padding - k_w) // stride + 1
    output = np.empty((out_channels, h_out, w_out))
    for channel in range(out_channels):
        for i in range(h_out):
            for j in range(w_out):
                patch = padded[
                    :,
                    i * stride : i * stride + k_h,
                    j * stride : j * stride + k_w,
                ]
                output[channel, i, j] = np.sum(patch * filters[channel]) + bias[channel]
    return output.round(4).tolist()