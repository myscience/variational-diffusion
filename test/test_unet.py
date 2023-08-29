import torch
import unittest

from torch import Tensor

from src.unet import UNet

class UNetTest(unittest.TestCase):
    def setUp(self) -> None:
        
        chn_dim = 3
        h = w = 16

        self.n_heads = 4
        self.emb_dim = 32

        self.img_h = h
        self.img_w = w
        self.chn_dim = chn_dim
        self.batch_size = 2

        self.input_shape_img = (self.batch_size, chn_dim, h, w)

        self.time = torch.rand(self.batch_size)

    def test_forward_with_img_attn(self):

        out_dim = 6
        out_shape = (self.batch_size, out_dim, self.img_h, self.img_w)

        unet = UNet(
            net_dim=4,
            out_dim=out_dim,
            ctrl_dim=3,
            use_cond=True,
            use_attn=True,
            num_group=4,
            adapter='b c h w -> b (h w) c',
        )


        inp_img = torch.randn(self.input_shape_img)
        inp_ctx = torch.randn(self.input_shape_img)

        out_tensor = unet(
            inp_img,
            self.time,
            ctrl=inp_ctx,
        )

        self.assertEqual(out_tensor.shape, out_shape)

    def test_forward_with_fourier(self):

        out_dim = 6
        out_shape = (self.batch_size, out_dim, self.img_h, self.img_w)

        unet = UNet(
            net_dim=4,
            out_dim=out_dim,
            adapter='b c h w -> b (h w) c',
            ctrl_dim=3,
            use_cond=True,
            use_attn=True,
            n_fourier=(7, 9, 1),
            num_group=4,
        )


        inp_img = torch.randn(self.input_shape_img)
        inp_ctx = torch.randn(self.input_shape_img)

        out_tensor = unet(
            inp_img,
            self.time,
            ctrl=inp_ctx,
        )

        self.assertEqual(out_tensor.shape, out_shape)

    