import numpy as np

def activation_functions(x: float, activation: str) -> list:
    """
    Returns the activation value and analytical derivative.
    """
    tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    if activation == "relu":
        if x > 0:
            return x, 1
        else:
            return 0, 0
    elif activation == "leaky_relu":
        if x >0:
            return x, 1
        else:
            return 0.01*x, 0.01
    elif activation == "sigmoid":
        s = sigmoid(x)
        return s, s*(1-s)
    elif activation == "tanh":
        t = tanh(x)
        return t, 1-t**2
    elif activation == "gelu":
        u = np.sqrt(2/np.pi)*(x+0.044715*x**3)
        t = tanh(u)
        gelu = 0.5*x*(1 + t)
        gelu_ = 0.5*(1+t) + 0.5*x*(1-t**2)*np.sqrt(2/np.pi)*(1+3*0.044715*x**2)
        return gelu, gelu_
    elif activation == "swish":
        s = sigmoid(x)
        return x*s, s + x*s*(1-s)