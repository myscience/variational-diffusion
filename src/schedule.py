import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Tuple

from itertools import pairwise

from .module.monotonic import MonotonicLinear

class LinearSchedule(nn.Module):
    '''
        Simple Linear schedule with learnable endpoints.
    '''

    def __init__(
        self,
        gamma_min : float = -13.3,
        gamma_max : float = 5.0,
    ) -> None:
        super().__init__()

        self.q : Tensor = nn.Parameter(torch.tensor(gamma_min))
        self.m : Tensor = nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t : float) -> Tensor:
        return self.m.abs() * t + self.q

class LearnableSchedule(nn.Module):
    '''
        Monotonic schedule represented by a MLP that
        learns the optimal schedule to follow to minimize
        the VLB variance (granting faster more stable training).
        
        Monotonicity is ensured by using MonotonicLinear layers
        as introduced in:
        `Constrained Monotonic Neural Networks` ICML (2023).
    '''

    def __init__(
        self,
        hid_dim : int | List[int] = 3, 
        gate_func : str = 'relu',
        act_weight : Tuple[float, float, float] = (7, 7, 2),   
    ) -> None:
        super().__init__()

        if isinstance(hid_dim, int): hid_dim = [hid_dim]
        dims = [1, *hid_dim, 1]

        # Create the MLP
        self.layers = nn.Sequential(
            *(MonotonicLinear(
                inp_dim, out_dim, bias=True,
                gate_func=gate_func,
                indicator=+1 if layer > 0 else -1,
                act_weight=act_weight,
            ) for layer, (inp_dim, out_dim) in enumerate(pairwise(dims)))
        )

    def forward(self, x : Tensor) -> Tensor:
        return self.layers(x)