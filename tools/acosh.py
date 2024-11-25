from typing import Any, Tuple, Optional

import torch
from torch import Tensor

eps = 1e-7

def sqrt(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=1e-9)  # Smaller epsilon due to precision around x=0.
    return torch.sqrt(x)

class LeakyClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        ctx.save_for_backward(x.ge(min) * x.le(max))
        return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None


def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)


def lorentz_product(x: Tensor, y: Tensor, keepdim: bool = False, dim: int = -1) -> Tensor:
    try:
        m = x * y
    except:
        m = torch.mm(x,y)
    if keepdim:
        ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
    else:
        ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
    return ret

def lorentz_norm(x: Tensor, **kwargs: Any) -> Tensor:
    product = lorentz_product(x, x, **kwargs)
    ret = sqrt(product)
    return ret


class Acosh(torch.autograd.Function):
    """
    Numerically stable arccosh that never returns NaNs.
    Returns acosh(x) = arccosh(x) = log(x + sqrt(max(x^2 - 1, eps))).
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        x = clamp(x, min=1 + eps)
        z = sqrt(x * x - 1.)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        z, = ctx.saved_tensors
        # z_ = clamp(z, min=eps)
        z_ = z
        return grad_output / z_

def acosh(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable arccosh that never returns NaNs.
    :param x: The input tensor.
    :return: log(x + sqrt(max(x^2 - 1, eps))
    """
    return Acosh.apply(x)