import torch

import torch.nn as nn

from .utils import default
from .schedule import LinearSchedule

class VariationalDiffusion(nn.Module):
    '''
    
    '''

    def __init__(
        self,
        backbone : nn.Module,
        schedule : nn.Module | None = None,                
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.schedule = default(schedule, LinearSchedule())

    