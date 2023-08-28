
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List

from .utils import exists
from .utils import default

from .module.conv import Upscale
from .module.conv import Downscale
from .module.conv import ContextRes
from .module.embedding import TimeEmbedding
from .module.embedding import FourierEmbedding
from .module.attention import AdaptiveAttention

class UNet(nn.Module):
    '''
        U-Net model as introduced in:
        "U-Net: Convolutional Networks for Biomedical Image Segmentation".
        It is a common choice as network backbone for diffusion models.         
    '''

    def __init__(
        self,
        net_dim : int = 4,
        out_dim : int | None = None,
        inp_chn : int = 3,
        dropout : float = 0.,
        adapter : str | Tuple[str, ...] = 'b c h w -> b (h w) c', 
        attn_dim : int = 128,
        ctrl_dim : int | None = None,
        use_cond : bool = False,
        use_attn : bool = False,
        chn_mult : List[int] = (1, 2, 4, 8),
        n_fourier : Tuple[int, ...] | None = None,
        num_group : int = 8,
        num_heads : int = 4,
    ) -> None:
        super().__init__()

        out_dim = default(out_dim, inp_chn)

        self.inp_chn = inp_chn
        self.use_cond = use_cond
        self.use_attn = use_attn

        # * Build the input embeddings
        # Optional Fourier Feature Embeddings
        self.fourier_emb = FourierEmbedding(*n_fourier) if exists(n_fourier) else nn.Identity()

        # Time Embeddings
        ctx_dim = net_dim * 4
        self.time_emb = nn.Sequential(
            TimeEmbedding(net_dim),
            nn.Linear(net_dim, ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, ctx_dim)
        )

        # NOTE: We need channels * 2 to accommodate for the self-conditioning
        tot_chn = inp_chn * (1 + use_cond + (2 * self.fourier_emb.n_feat if exists(n_fourier) else 0))

        self.proj_inp = nn.Conv2d(tot_chn, net_dim, 7, padding = 3)

        dims = [net_dim, *map(lambda m: net_dim * m, chn_mult)]
        mid_dim = dims[-1]

        dims = list(zip(dims, dims[1:]))

        # * Building the model. It has three main components:
        # * 1) The downscale modules
        # * 2) The bottleneck modules
        # * 3) The upscale modules
        self.downs = nn.ModuleList([])
        self.ups   = nn.ModuleList([])
        num_resolutions = len(dims)

        # Build up the downscale module part
        for idx, (dim_in, dim_out) in enumerate(dims):
            is_last = idx >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ContextRes(dim_in, dim_in, ctx_dim=ctx_dim, num_group=num_group, dropout=dropout),
                ContextRes(dim_in, dim_in, ctx_dim=ctx_dim, num_group=num_group, dropout=dropout),
                AdaptiveAttention(attn_dim, num_heads, adapter, qry_dim=dim_in, key_dim=ctrl_dim) if use_attn else nn.Identity(),
                nn.Conv2d(dim_in, dim_out, 3, padding = 1) if is_last else Downscale(dim_in, dim_out)
            ]))

        # Buildup the bottleneck module
        self.mid_block1 = ContextRes(mid_dim, mid_dim, ctx_dim=ctx_dim, num_group=num_group)
        self.mid_attn   = AdaptiveAttention(attn_dim, num_heads, adapter, qry_dim=mid_dim, key_dim=ctrl_dim)
        self.mid_block2 = ContextRes(mid_dim, mid_dim, ctx_dim=ctx_dim, num_group=num_group)

        # Build the upscale module part
        # NOTE: We need to make rooms for incoming residual connections from the downscale layers
        for idx, (dim_in, dim_out) in enumerate(reversed(dims)):
            is_last = idx >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ContextRes(dim_in + dim_out, dim_out, ctx_dim=ctx_dim, num_group=num_group, dropout=dropout),
                ContextRes(dim_in + dim_out, dim_out, ctx_dim=ctx_dim, num_group=num_group, dropout=dropout),
                AdaptiveAttention(attn_dim, num_heads, adapter, qry_dim=dim_out, key_dim=ctrl_dim) if use_attn else nn.Identity(),
                nn.Conv2d(dim_out, dim_in, 3, padding = 1) if is_last else Upscale(dim_out, dim_in)
            ]))

        self.final = ContextRes(net_dim * 2, net_dim, ctx_dim = ctx_dim, num_group = num_group)
        self.proj_out = nn.Conv2d(net_dim, out_dim, 1)

    def forward(
        self,
        imgs : Tensor,
        time : Tensor,
        cond : Tensor | None = None,
        ctrl : Tensor | None = None,
    ) -> Tensor:
        '''
            Compute forward pass of the U-Net module. Expect input
            to be image-like and expects an auxiliary time signal
            (1D-like) to be provided as well. An optional contextual
            signal can be provided and will be used by the attention
            gates that will function as cross-attention as opposed
            to self-attentions.

            Params:
                - imgs: Tensor of shape [batch, channel, H, W]
                - time: Tensor of shape [batch, 1]
                - context[optional]: Tensor of shape [batch, seq_len, emb_dim]

            Returns:
                - imgs: Processed images, tensor of shape [batch, channel, H, W]
        '''

        # Optional self-conditioning to the model (we default to original
        # input size before fourier embeddings are added)
        cond = default(cond, torch.zeros_like(imgs))

        # Add (optional) Fourier Embeddings
        imgs = self.fourier_emb(imgs)

        if self.use_cond: imgs = torch.cat((imgs, cond), dim = 1)

        x : Tensor = self.proj_inp(imgs)
        t : Tensor = self.time_emb(time)

        h = [x.clone()]

        for conv1, conv2, attn, down in self.downs:
            x = conv1(x, t)
            h += [x]

            x = conv2(x, t)
            x = attn(x, ctrl) if self.use_attn else x
            h += [x]

            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, ctrl)
        x = self.mid_block2(x, t)

        for conv1, conv2, attn, up in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = conv1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = conv2(x, t)
            x = attn(x, ctrl) if self.use_attn else x

            x = up(x)

        x = torch.cat((x, h.pop()), dim = 1)

        x = self.final(x, t)

        return self.proj_out(x)