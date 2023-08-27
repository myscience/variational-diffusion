import torch
import unittest

from torch import Tensor

from src.module.attention import AdaptiveAttention

class AdaptiveAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        
        batch_size = 2
        chn_dim = 8
        h = w = 16
        l = h * w

        self.n_heads = 4
        self.emb_dim = 32

        self.img_h = h
        self.img_w = w
        self.chn_dim = chn_dim

        self.input_shape_txt = (batch_size, l, chn_dim)
        self.input_shape_mp3 = (batch_size, chn_dim, l)
        self.input_shape_img = (batch_size, chn_dim, h, w)
        self.input_shape_mp4 = (batch_size, chn_dim, h, w, l)


    def test_forward_self_attn_image(self):

        attn = AdaptiveAttention(
            emb_dim=self.emb_dim,
            n_heads=self.n_heads,

            # Test with adapter just for the query, it then expects
            # key|val to be already sequence-like 
            pattern='b c h w -> b (h w) c',
            qry_dim=self.chn_dim,
            batch_first=True,
        )

        qry : Tensor = torch.randn(self.input_shape_img)
        key : Tensor = torch.randn(self.input_shape_img)
        val : Tensor = torch.randn(self.input_shape_img)

        out_tensor, _ = attn(qry, key, val)

        self.assertEqual(out_tensor.shape, self.input_shape_img)

    def test_forward_attn_img_txt(self):

        attn = AdaptiveAttention(
            emb_dim=self.emb_dim,
            n_heads=self.n_heads,

            # Test with adapter just for the query, it then expects
            # key|val to be already sequence-like 
            pattern=(
                'b c h w -> b (h w) c',
                '... -> ...',
                '... -> ...',
            ),
            qry_dim=self.chn_dim,
            batch_first=True,
        )

        qry : Tensor = torch.randn(self.input_shape_img)
        key : Tensor = torch.randn(self.input_shape_txt)
        val : Tensor = torch.randn(self.input_shape_txt)

        out_tensor, _ = attn(qry, key, val)

        self.assertEqual(out_tensor.shape, self.input_shape_img)