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

def relu(x: list) -> list:
    return np.maximum(0, x)


def resnet_block(x: list, W1: list, b1: list, W2: list, b2: list) -> dict:
    """
    Returns the two convolution results and residual output.
    """
    x = np.array(x, dtype=np.float64)

    conv1 = conv2d(x, W1, b1, padding=1)
    conv1 = relu(conv1)
    conv2 = conv2d(conv1, W2, b2, padding=1)
    output = relu(conv2 + x)
    return {
        "conv1": conv1,
        "conv2": conv2,
        "output": output,
    }
