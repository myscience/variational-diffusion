import torch

import torch.nn as nn
from torch import Tensor
from torch import sqrt, sigmoid
from torch import exp, expm1
from torch import log_softmax
from typing import Tuple
from itertools import pairwise

from tqdm.auto import tqdm

from einops import rearrange

from .utils import default
from .schedule import LinearSchedule

class VariationalDiffusion(nn.Module):
    '''
    
    '''

    def __init__(
        self,
        backbone : nn.Module,
        schedule : nn.Module | None = None,  
        img_shape : Tuple[int, int] = (64, 64),
        vocab_size : int = 256,              
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.schedule = default(schedule, LinearSchedule())

        img_chn = self.backbone.inp_chn

        self.img_shape = (img_chn, *img_shape)
        self.vocab_size = vocab_size

    @property
    def device(self):
        return next(self.backbone.parameters()).device
    

    @torch.no_grad()
    def forward(
        self,
        num_imgs : int,
        num_steps : int,
        seed_noise : Tensor | None = None,
    ) -> Tensor:
        '''
            We reserve the forward call of the model to the posterior sampling,
            that is used with a fully trained model to generate samples, hence
            the torch.no_grad() decorator.
        '''
        device = self.device

        z_s = default(seed_noise, torch.randn(num_imgs, *self.img_shape), device=device)

        # Sample the reverse time-steps and compute the corresponding
        # noise schedule values (gammas)
        time = torch.linspace(1., 0., num_steps + 1, device=device)
        gamma = self.schedule(time)

        for gamma_t, gamma_s in tqdm(pairwise(gamma), total=num_steps):
            # Sample from the backward diffusion process
            z_s = self._coalesce(z_s, gamma_t, gamma_s)

        # After the backward process we are left with the z_0 latent from
        # which we should estimate the probabilities of the data x via
        # p (x | z_0) =~ q (z_0 | x), which is a good approximation whenever
        # SNR(t = 0) is high-enough (as we basically don't corrupt x).
        # NOTE: We pass a rescaled z_0 by the mean which is alpha_0,
        #       we re-use gamma_s as the last step corresponds to t=0
        alpha_0 = sqrt(sigmoid(gamma_s))

        # Decode the probability for each data bin, expected prob shape is:
        # [batch_size, C, H, W, vocal_size]
        prob = self._data_prob(z_s / alpha_0, gamma_s)

        # Our sample is obtained by taking the highest probability bin among
        # all the possible data values
        img = torch.argmax(prob, dim=-1) 

        # Normalize image to be in [0, 1]
        return img.float() / (self.vocab_size - 1)
    
    def continuous_loss(self, imgs : Tensor) -> Tensor:
        '''
            Compute the L_∞ loss (T -> ∞), which is composed of three terms:

                L_∞ = L_diffusion + L_latent + L_reconstruction.

            This loss comes from minimizing the Variational Lower Bound (VLB),
            which is: -log p(x) < -VLB(x). 
        '''

        # Rescale image tensor (expected in range [0, 1]) to [-1 + 1/vs, +1 - 1/vs]
        # (vs = vocab-size)

        

    def _diffuse(self, x_0 : Tensor, gamma_t : Tensor) -> Tensor:
        '''
            Forward diffusion: we sample from q(z_t | x_0). This is
            easy sampling as we only need to sample from a standard
            normal with known SNR(t). We have:
                q(z_t | x_0) = N(alpha_t x_0 | sigma_t**2 * I), with

                SNR(t) = alpha_t ** 2 / sigma_t ** 2

            NOTE: Time is effectively parametrized via the SNR which
                  in turn is computed via the noise schedule that can
                  either be linear of a monotonic network.
        '''

        # Compute the alpha_t and sigma_t using the noise schedule

        alpha_t = sqrt(sigmoid(-gamma_t))
        sigma_t = sqrt(sigmoid(+gamma_t))

        noise = torch.randn_like(x_0)

        return alpha_t * x_0 + sigma_t * noise
    
    def _coalesce(self, z_t : Tensor, gamma_t : Tensor, gamma_s : Tensor) -> Tensor:
        '''
            Backward diffusion: we sample from p(z_s | z_t, x = x_theta(z_t ; t)).
            This is a bit more involved as we need to sample from a
            distribution that depends on the previous sample. We have:
                p(z_s | z_t, x = x_theta(z_t ; t)) = N(mu_theta, sigma_Q ** 2 * I),
            
            where we eventually have (see Eq.(32-33) in Appendix A of paper):

                mu_theta = alpha_s / alpha_t * (z_t + sigma_t * expm1(gamma_s - gamma_t))\
                            * eps_theta(z_t ; t)

                sigma_Q = sigma_s ** 2 * (-expm1(gamma_s - gamma_t))
        '''

        alpha_s_sq = sigmoid(-gamma_s)
        alpha_t_sq = sigmoid(-gamma_t)

        sigma_t = sqrt(sigmoid(+gamma_t))
        c = -expm1(gamma_s - gamma_t)

        # Predict latent noise eps_theta using backbone model
        eps = self.backbone(z_t, gamma_t) # NOTE: We should add here conditioning if needed

        # Compute new latent z_s mean and std
        scale = sqrt((1 - alpha_s_sq) * c) 
        mean = sqrt(alpha_s_sq / alpha_t_sq) * (z_t - c * sigma_t * eps)

        return mean + scale * torch.randn_like(z_t)
    
    def _data_prob(self, z_0 : Tensor, gamma_0 : Tensor) -> Tensor:
        '''
            Compute the probability distribution for p(x | z_0). This distribution is
            approximated by p(x | z_0) ~ Prod_i q(z_0(i) | x(i)), which is sensible
            whenever SNR(t=0) is high enough. Here q(z_0(i) | x(i)) represent the
            latent code for pixel i-th given the original pixel value. The logits
            are estimated as:
            
                    -1/2 SNR(t=0) * (z_0 / alpha_0 - k) ** 2,
            
            where k takes all possible data values (hence k take vocab_size values).

            NOTE: We assume input z_0 has already been normalized, so we actually
                  expect z_0 / alpha_0. Moreover, we have: SNR(t=0) = exp(-gamma_0)
        '''

        # Add vocab_size dimension
        z_0 = rearrange(z_0, '... -> ... 1')

        x_lim = 1 - 1 / self.vocab_size
        x_val = torch.linspace(-x_lim, +x_lim, self.vocab_size, device = self.device)

        logits = -.5 * exp(-gamma_0) * (z_0 - x_val) ** 2

        # Normalize along the vocab_size dimension
        return log_softmax(logits, dim=-1)


 




