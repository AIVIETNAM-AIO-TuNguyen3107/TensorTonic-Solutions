import numpy as np

def categorical_encode(data, method="label"):
    """
    Returns: encoded result based on method
    """
    data = np.array(data)
    classes = np.unique(data)
    k = len(classes)
    encoded_values = np.arange(k)
    encoded_value_by_class = {
        key: value
        for key, value in zip(classes, encoded_values, strict=True)
    }
    encoded = np.array([ encoded_value_by_class[key] for key in data ])
    if method == "label":
        return { "encoded": encoded, "classes": classes }
    elif method == "onehot":
        one_hot = np.zeros((len(data), k), dtype=int)
        one_hot[np.arange(len(data)), encoded] = 1
        return one_hot