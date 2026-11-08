import torch
import torch.nn.functional as F

def composite_layer(x: torch.Tensor, bn_gamma: torch.Tensor, bn_beta: torch.Tensor,
                    bn_mean: torch.Tensor, bn_var: torch.Tensor,
                    conv_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 output of the DenseNet composite transformation.
    """
    bn_gamma = bn_gamma[None, :, None, None]
    bn_beta = bn_beta[None, :, None, None]
    bn_mean = bn_mean[None, :, None, None]
    bn_var = bn_var[None, :, None, None]

    normalized = bn_gamma*(x-bn_mean)/torch.sqrt(bn_var+eps) + bn_beta

    return F.conv2d(F.relu(normalized), conv_weight, padding=1, bias=None)