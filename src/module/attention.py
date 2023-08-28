import torch.nn as nn

from torch.nn import MultiheadAttention

from itertools import starmap

from einops import rearrange
from typing import Tuple
from torch import Tensor

from ..utils import default

class Adapter(nn.Module):
    def __init__(
        self,
        pattern : str | Tuple[str, ...],
        qry_dim : int,
        key_dim : int | None = None,
        val_dim : int | None = None,
        emb_dim : int | None = None,
    ) -> None:
        super(Adapter, self).__init__()

        # If no adapter was provided for key and values, we
        # assume self-attention is going to be computed, so
        # we simply replicate the pattern for both key and values
        if isinstance(pattern, str): pattern = [pattern] * 3
        if len(pattern) == 2: pattern = (*pattern, pattern[-1])

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, qry_dim)
        emb_dim = default(emb_dim, qry_dim)

        self.pattern = pattern
        self.emb_dim = emb_dim

        self.inv_pattern = ' -> '.join(pattern[0].split('->')[::-1])

        self.to_q = self._get_adapter(pattern[0], chn_inp=qry_dim, chn_out=emb_dim)
        self.to_k = self._get_adapter(pattern[1], chn_inp=key_dim, chn_out=emb_dim)
        self.to_v = self._get_adapter(pattern[2], chn_inp=val_dim, chn_out=emb_dim)

        self.from_q = self._get_adapter(pattern[0], chn_inp=emb_dim, chn_out=qry_dim)

    def forward(
        self,
        qry : Tensor,
        key : Tensor | None = None,
        val : Tensor | None = None,
    ) -> Tensor:
        '''
        '''

        q = self.to_q(qry)
        k = self.to_k(key)
        v = self.to_v(val)

        self.q_shape = q.shape

        q, k, v = starmap(lambda t, adapt : rearrange(t, adapt), zip((q, k, v), self.pattern))

        return q, k, v
    
    def restore(self, attn : Tensor) -> Tensor:
        '''
        '''

        if not hasattr(self, 'q_shape'):
            raise ValueError('Cannot restore before forward pass has been called')

        # Prepare the appropriate kwargs for rearrange by composing the known
        # inverse adapter with the stored qry shape from forward pass
        names = [c for c in self.inv_pattern.split('->')[-1] if c.isalpha()]
        kwargs = {k : v for k, v in zip(names, self.q_shape)}

        attn = rearrange(attn, self.inv_pattern, **kwargs)
    
        return self.from_q(attn)

    def _get_adapter(
        self,
        pattern : str | Tuple[str, ...],
        chn_inp : int,
        chn_out : int,
    ) -> nn.Module:
        if chn_inp == chn_out: return nn.Identity()

        dim_out = sum([c.isalpha() for c in pattern.split('->')[-1]])

        match dim_out:
            case 0: return nn.Linear(chn_inp, chn_out, bias=False)
            case 3: return nn.Conv1d(chn_inp, chn_out, 1, bias=False)
            case 4: return nn.Conv2d(chn_inp, chn_out, 1, bias=False)
            case 5: return nn.Conv3d(chn_inp, chn_out, 1, bias=False)
            case _: pass

        raise ValueError(f'Input shape not supported. Got {dim_out}')

class AdaptiveAttention(MultiheadAttention):
    def __init__(
        self,
        emb_dim,
        n_heads,
        pattern : str,
        qry_dim : int | None = None,
        key_dim : int | None = None,
        val_dim : int | None = None,
        batch_first : bool = True,
        **kwargs
    ) -> None:
        super(AdaptiveAttention, self).__init__(emb_dim, n_heads, batch_first=batch_first, **kwargs)

        qry_dim = default(qry_dim, emb_dim)
        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)
        
        # Build the attention adepter
        self.adapter = Adapter(
            pattern=pattern,
            qry_dim=qry_dim,
            key_dim=key_dim,
            val_dim=val_dim,
            emb_dim=emb_dim,
        )

    def forward(
        self,
        qry : Tensor,
        key : Tensor | None = None,
        val : Tensor | None = None,
        return_weights : bool = False,
        **kwargs
    ) -> Tensor | Tuple[Tensor, Tensor]:
        '''
        
        '''

        key = default(key, qry)
        val = default(val, key)

        # Adapt the inputs to the expected format by the MHA module
        qry, key, val = self.adapter(qry, key, val)

        # Compute the attention output
        attn, attn_weights = super().forward(qry, key, val, **kwargs)

        # Restore the correct output format
        attn = self.adapter.restore(attn)

        return (attn, attn_weights) if return_weights else attn
