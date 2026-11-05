import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through residual Jacobians.
    """
    result = np.asarray(x, dtype=np.float64).copy()
    n = result.shape[0]
    J = np.eye(n)
    for grad in gradients_F:
        J @= np.eye(n) + np.asarray(grad, dtype=np.float64)
    return result@J

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through plain Jacobians.
    """
    result = np.asarray(x, dtype=np.float64).copy()
    n = result.shape[0]
    J = np.eye(n)
    for grad in gradients_F:
        J @= np.asarray(grad, dtype=np.float64)
    return result@J