import torch
from torch import Tensor


def sig(t: Tensor) -> Tensor:
    return torch.reciprocal(1 + torch.exp(-1 * t))


def tanh(t: Tensor) -> Tensor:
    exp_t = torch.exp(t)
    exp_neg_t = torch.exp(-1 * t)
    return (exp_t - exp_neg_t) * torch.reciprocal(exp_t + exp_neg_t)


def relu(t: Tensor) -> Tensor:
    """ Rectifier activation function.
        The relu function of a tensor T is the element-wise max between 0 and the appropriate 
        element of T.
    """
    tensor = t if torch.is_tensor(t) else torch.tensor(t)
    return torch.max(tensor, torch.zeros_like(tensor))


def swish(t: Tensor, beta: float) -> Tensor:
    """ Swish activation function proposed by Ramachandran et al. on their paper "Searching for
        Activation Functions" (arXiv:1710.05941v2).
        The Swish function of a tensor T is defined as: T * sigmoid(beta * T).
    """
    tensor = t if torch.is_tensor(t) else torch.tensor(t)
    beta_tensor = torch.full_like(tensor, beta)
    return tensor * sig(beta_tensor * tensor)


def celu(t: Tensor, alpha: float) -> Tensor:
    """ Continuously Differentiable Exponential Linear Units function as proposed by Barron on his
        paper "Continuously Differentiable Exponential Linear Units" (arXiv:1704.07483).
        The CELU function of a tensor T is:
            - T[i] when T[i] >= 0
            - alpha * (exp(T[i] / alpha) - 1)
        for each element i of the tensor T.
    """
    tensor = t if torch.is_tensor(t) else torch.tensor(t)
    zero_tensor = torch.zeros_like(tensor)
    alpha_tensor = torch.full_like(tensor, alpha)
    return torch.max(zero_tensor, tensor) + torch.min(zero_tensor, alpha_tensor * (
                torch.exp(tensor / alpha_tensor) - torch.full_like(tensor, 1)))
