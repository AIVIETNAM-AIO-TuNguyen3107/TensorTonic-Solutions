import torch
import torch.nn.functional as F

def bottleneck_layer(x: torch.Tensor, bn1_gamma: torch.Tensor, bn1_beta: torch.Tensor,
                     bn1_mean: torch.Tensor, bn1_var: torch.Tensor,
                     conv1_weight: torch.Tensor, bn2_gamma: torch.Tensor,
                     bn2_beta: torch.Tensor, bn2_mean: torch.Tensor,
                     bn2_var: torch.Tensor, conv2_weight: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 output of the DenseNet-B bottleneck transformation.
    """
    bn1_gamma = bn1_gamma[None, :, None, None]
    bn1_beta = bn1_beta[None, :, None, None]
    bn1_mean = bn1_mean[None, :, None, None]
    bn1_var = bn1_var[None, :, None, None]
    
    bn2_gamma = bn2_gamma[None, :, None, None]
    bn2_beta = bn2_beta[None, :, None, None]
    bn2_mean = bn2_mean[None, :, None, None]
    bn2_var = bn2_var[None, :, None, None]

    normalized = bn1_gamma*(x-bn1_mean)/torch.sqrt(bn1_var+eps) + bn1_beta
    conv1 = F.conv2d(F.relu(normalized), conv1_weight, bias=None, padding=0)
    normalized = bn2_gamma*(conv1-bn2_mean)/torch.sqrt(bn2_var+eps) + bn2_beta
    conv2 = F.conv2d(F.relu(normalized), conv2_weight, padding=1, bias=None)
    return conv2