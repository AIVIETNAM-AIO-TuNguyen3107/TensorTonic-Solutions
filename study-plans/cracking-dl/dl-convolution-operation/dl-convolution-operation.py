import numpy as np


def conv2d(
    input: list, filters: list, bias=None, padding: int = 0, stride: int = 1
) -> list:
    """
    Returns the channel-first cross-correlation output.
    """
    input = np.array(input, dtype=np.float64)
    _, h, w = input.shape
    if padding != 0:
        pad_width = [(0, 0)] * input.ndim
        pad_width[-1] = (padding, padding)
        pad_width[-2] = (padding, padding)

        input = np.pad(input, pad_width, mode="constant", constant_values=0)
    _, h_padded, w_padded = input.shape
    output = []
    for filter in filters:
        filter = np.array(filter, dtype=np.float64)
        _, k_h, k_w = filter.shape
        if bias is not None:
            bias = np.array(bias, dtype=np.float64)
        else:
            bias = np.zeros((1))
        for i in range(0, h_padded, stride):
            for j in range(0, w_padded, stride):
                if i + k_h > h_padded or j + k_w > w_padded:
                    continue
                output.append(
                    np.sum(input[:, i : i + k_h, j : j + k_w] * filter) + bias
                )
    h_out = (h + 2 * padding - k_h) // stride + 1
    return list(np.array(output).reshape((len(filters), h_out, h_out)))