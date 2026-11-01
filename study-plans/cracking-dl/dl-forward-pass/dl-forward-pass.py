import numpy as np

def forward_pass(x: list, weights: list, biases: list) -> dict:
    """
    Returns all activations and pre-activations.
    """

    activations = [x]
    pre_activations = []
    x = np.array(x, dtype=np.float64)
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        weight = np.array(weight, dtype=np.float64)
        bias = np.array(bias, dtype=np.float64)
        pre_activation = weight@activations[-1] + bias
        if index == len(weights)-1:
            activation = pre_activation
        else:
            activation = np.where(pre_activation>0, pre_activation, 0)
        activations.append(list(activation))
        pre_activations.append(list(pre_activation))
    return {
        "activations": activations,
        "pre_activations": pre_activations
    }