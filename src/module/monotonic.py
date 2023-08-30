import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, Callable

from einops import repeat
from functools import partial

from ..utils import default

def saturating_func(
    x : Tensor,
    conv_f : Callable[[Tensor], Tensor] = None,
    conc_f : Callable[[Tensor], Tensor] = None,
    slope : float = 1.,
    const : float = 1.,
) -> Tensor:
    conv = conv_f(+torch.ones_like(x) * const)

    return slope * torch.where(
        x <= 0,
        conv_f(x + const) - conv,
        conc_f(x - const) + conv,
    )

class MonotonicLinear(nn.Linear):
    '''
        Monotonic Linear Layer as introduced in:
        `Constrained Monotonic Neural Networks` ICML (2023).

        Code is a PyTorch implementation of the official repository:
        https://github.com/airtai/mono-dense-keras/
    '''

    def __init__(
        self,
        in_features  : int,
        out_features : int,
        bias : bool = True,
        gate_func : str = 'elu',
        indicator : int | Tensor | None = None,
        act_weight : str | Tuple[float, float, float] = (7, 7, 2),
    ) -> None:
        # Assume positive monotonicity in all input features
        indicator = default(indicator, torch.ones(in_features))

        if isinstance(indicator, int):
            indicator = torch.ones(in_features) * indicator

        assert indicator.dim() == 1, 'Indicator tensor must be 1-dimensional.'
        assert indicator.size(-1) == in_features, 'Indicator tensor must have the same number of elements as the input features.'
        assert len(act_weight) == 3, f'Relative activation weights should have len = 3. Got {len(act_weight)}'
        if isinstance(act_weight, str): assert act_weight in ('concave', 'convex')

        self.indicator = indicator

        # Compute the three activation functions: concave|convex|saturating
        match gate_func:
            case 'elu' : self.act_conv = F.elu
            case 'silu': self.act_conv = F.silu
            case 'gelu': self.act_conv = F.gelu
            case 'relu': self.act_conv = F.relu
            case 'selu': self.act_conv = F.selu
            case _: raise ValueError(f'Unknown gating function {gate_func}')

        self.act_conc = lambda t : -self.act_conv(-t)
        self.act_sat = partial(
            saturating_func,
            conv_f=self.act_conv,
            conc_f=self.act_conc,
        )

        match act_weight:
            case 'concave': self.act_weight = torch.tensor((1, 0, 0))
            case 'convex' : self.act_weight = torch.tensor((0, 1, 0))
            case _: self.act_weight = torch.tensor(act_weight) / sum(act_weight)

        # Build the layer weights and bias
        super(MonotonicLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x : Tensor) -> Tensor:
        '''
        '''

        # Get the absolute values of the weights
        abs_weights = self.weight.data.abs()

        # * Use monotonicity indicator T to adjust the layer weights
        # * T_i = +1 -> W_ij <=  || W_ij ||
        # * T_i = -1 -> W_ij <= -|| W_ij ||
        # * T_i =  0 -> do nothing
        mask_pos = self.indicator == +1
        mask_neg = self.indicator == -1

        self.weight.data[..., mask_pos] = +abs_weights[..., mask_pos]
        self.weight.data[..., mask_neg] = -abs_weights[..., mask_neg]

        # Get the output of linear layer
        out = super().forward(x)

        # Compute output by adding non-linear gating according to
        # relative importance of activations
        s_conv, s_conc, _ = (self.act_weight * self.out_features).round()
        s_conv = int(s_conv)
        s_conc = int(s_conc)
        s_sat = self.out_features - s_conv - s_conc

        i_conv, i_conc, i_sat = torch.split(
            out, (s_conv, s_conc, s_sat), dim=-1
        )

        out = torch.cat((
                self.act_conv(i_conv),
                self.act_conc(i_conc),
                self.act_sat (i_sat),
            ),
            dim=-1,
        )

        return out