import torch
import torch.nn as nn

from torch import Tensor

from einops import einsum
from einops import rearrange

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
        time *= 1000

        # Check for correct time shape
        bs, _ = time.shape

        half_dim = self.emb_dim // 2        
        emb_time = torch.empty((bs, self.emb_dim), device = time.device)

        pos_n = torch.arange(half_dim, device = time.device)
        inv_f = 1. / (self.base ** (pos_n / (half_dim - 1)))

        emb_v = einsum(time, inv_f, 'b _, f -> b f')

        emb_time[..., 0::2] = emb_v.sin()
        emb_time[..., 1::2] = emb_v.cos()

        return emb_time
    
class FourierEmbedding(nn.Module):
    '''
        Set of Fourier Features to add to the input latent
        code "z" of the noise-predictor UNet model to ease
        its handling of the high-frequency components of the
        input, which have a significant impact on the likelihood
        (despite not that much for the visual appearance).
    '''

    def __init__(
        self,
        n_min : int = 7,
        n_max : int = 8,
        n_step : int = 1,
    ) -> None:
        '''
        
        '''
        super().__init__()

        self.n_exp = torch.arange(n_min, n_max, n_step)
        self.n_feat = len(self.n_exp)

    def forward(self, z : Tensor) -> Tensor:
        '''
            Add the Fourier features to the input latent code.
            The features are concatenated to the channel dimension
            of the input vector, which is expected to have shape
            [batch_size, chn_dim, ...].

            The Fourier features are defined as:
            - f^n_ijk = sin(2^n pi z_ijk)
            - g^n_ijk = cos(2^n pi z_ijk)
        '''

        (bs, chn, *_), device = z.shape, z.device

        freq = einsum(2 ** self.n_exp.to(device), z, 'n, b c ... -> b n c ...')
        freq = rearrange(freq, 'b n c ... -> b (n c) ...')

        f = freq.sin()
        g = freq.cos()

        return torch.cat([z, f, g], dim = 1)


