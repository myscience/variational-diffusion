import torch
import torch.nn as nn

from torch import Tensor

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
        
        Monotonicity is ensured by restricting net weights to
        be strictly positive.
    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)