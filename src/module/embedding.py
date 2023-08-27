import torch
import torch.nn as nn

from torch import Tensor

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
        # NOTE: We multiply by 1000 because time in variational models
        #       is formalized in [0, 1], so we just upscale to 1000 for
        #       proper embedding computation
        time = torch.atleast_1d(time) * 1000
        bs = len(time)

        half_dim = self.emb_dim // 2        
        emb_time = torch.empty((bs, self.emb_dim), device = time.device)

        pos_n = torch.arange(half_dim, device = time.device)
        inv_f = 1. / (self.base ** (pos_n / (half_dim - 1)))

        emb_v = torch.outer(time, inv_f)

        emb_time[..., 0::2] = emb_v.sin()
        emb_time[..., 1::2] = emb_v.cos()

        return emb_time