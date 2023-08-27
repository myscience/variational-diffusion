
import torch.nn as nn
from torch import Tensor

from ..utils import exists
from ..utils import default
from ..utils import enlarge_as

def Upscale(dim_in, dim_out : int = None, factor : int = 2):
    return nn.Sequential(
        nn.Upsample(scale_factor = factor, mode = 'nearest'),
        nn.Conv2d(dim_in, default(dim_out, dim_in), 3, padding = 1)
    ) if factor > 1 else nn.Identity()

def Downscale(dim_in, dim_out : int = None, factor : int = 2):
    return nn.Conv2d(dim_in, default(dim_out, dim_in), 2 * factor, factor, 1)\
        if factor > 1 else nn.Identity()

class ContextRes(nn.Module):
    '''
        Convolutional Residual Block with context embedding
        injection support, used by Diffusion Models. It is
        composed of two convolutional layers with normalization.
        The context embedding signal is injected between the two
        convolutions (optionally) and is added to the input to
        the second one.
    '''

    def __init__(
        self,
        inp_dim : int,
        out_dim : int | None = None,
        hid_dim : int | None = None,
        ctx_dim : int | None = None,
        num_group : int = 8,
        dropout : float = 0.,
    ) -> None:
        super().__init__()

        out_dim = default(out_dim, inp_dim)
        hid_dim = default(hid_dim, out_dim)
        ctx_dim = default(ctx_dim, out_dim)

        self.time_emb = nn.Sequential(
            nn.SiLU(inplace = False),
            nn.Linear(ctx_dim, hid_dim),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp_dim, hid_dim, kernel_size = 3, padding = 1),
            nn.GroupNorm(num_group, hid_dim),
            nn.SiLU(inplace = False),
        )

        self.conv2 = nn.Sequential(
            *([nn.Dropout(dropout)] * (dropout > 0.)),
            nn.Conv2d(hid_dim, out_dim, kernel_size = 3, padding = 1),
            nn.GroupNorm(num_group, out_dim),
            nn.SiLU(inplace = False),
        )

        self.skip = nn.Conv2d(inp_dim, out_dim, 1) if inp_dim != out_dim else nn.Identity()

    def forward(
        self,
        inp : Tensor,
        ctx : Tensor | None = None,
    ) -> Tensor:
        
        # Perform first convolution block
        h = self.conv1(inp)

        if exists(ctx):
            # Add embedded time signal with appropriate
            # broadcasting to match image-like tensors
            ctx = self.time_emb(ctx)
            h += enlarge_as(ctx, h)

        h = self.conv2(h)

        return self.skip(inp) + h