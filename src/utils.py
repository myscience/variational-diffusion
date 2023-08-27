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

class TimeEmbedding(nn.Module):
    '''
        Embedding for time-like data used by diffusion models.
    '''

    def __init__(
        self,
        emb_dim : int,
        base : int = 10000
    ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.base = base

    def forward(self, time : Tensor) -> Tensor:
        time = torch.atleast_1d(time)
        bs = len(time)

        half_dim = self.emb_dim // 2        
        emb_time = torch.empty((bs, self.emb_dim), device = time.device)

        pos_n = torch.arange(half_dim, device = time.device)
        inv_f = 1. / (self.base ** (pos_n / (half_dim - 1)))

        emb_v = torch.outer(time, inv_f)

        emb_time[..., 0::2] = emb_v.sin()
        emb_time[..., 1::2] = emb_v.cos()

        return emb_time