import torch
import torch.nn as nn

from typing import Any
from torch import Tensor

from einops import rearrange

def exists(var : Any | None) -> bool:
    return var is not None

def default(var : Any | None, val : Any) -> Any:
    return var if exists(var) else val

def enlarge_as(a : Tensor, b : Tensor) -> Tensor:
    '''
        Add sufficient number of singleton dimensions
        to tensor a **to the right** so to match the
        shape of tensor b. NOTE that simple broadcasting
        works in the opposite direction.
    '''
    return rearrange(a, f'... -> ...{" 1" * (b.dim() - a.dim())}').contiguous()