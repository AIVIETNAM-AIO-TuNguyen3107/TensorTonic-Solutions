import math

def weight_init_params(layer_dims: list, method: str) -> list:
    """
    Returns initialization metadata for each weight matrix.
    """
    result = []
    for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
        if method == "random_normal":
            scale = 1
        elif method == "xavier_normal":
            scale = math.sqrt(2/(fan_in+fan_out))
        elif method == "xavier_uniform":
            scale = math.sqrt(6/(fan_in+fan_out))
        elif method == "kaiming_normal":
            scale = math.sqrt(2/fan_in)
        elif method == "kaiming_uniform":
            scale = math.sqrt(6/fan_in)
        else:
            scale = math.sqrt(6/fan_in)
        init_metadata = {
            "fan_in": fan_in,
            "fan_out": fan_out,
            "scale": round(float(scale),4),
            "shape": [fan_out, fan_in]
        }
        result.append(init_metadata)
    return result