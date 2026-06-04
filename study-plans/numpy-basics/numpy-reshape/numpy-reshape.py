import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    array = np.array(data, dtype=np.float64)

    if operation=='flatten':
        return array.flatten()
    elif operation=='transpose':
        return array.T
    elif operation=='add_batch':
        return np.expand_dims(array, axis=0)