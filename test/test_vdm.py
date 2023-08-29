import torch
import unittest

from torch import Tensor

from src.unet import UNet
from src.vdm import VariationalDiffusion
from src.schedule import LinearSchedule

class VariationalDiffusionTest(unittest.TestCase):
    def setUp(self) -> None:
        
        chn_dim = 3
        h = w = 16

        self.n_heads = 4
        self.emb_dim = 32

        self.img_h = h
        self.img_w = w
        self.chn_dim = chn_dim
        self.batch_size = 2
        self.vocab_size = 256

        self.img_shape = (chn_dim, h, w)

    def test_forward(self):

        num_imgs = 4
        num_steps = 5

        vdm = VariationalDiffusion(
            backbone=UNet(
                net_dim=4,
                ctrl_dim=None,
                use_cond=False,
                use_attn=True,
                num_group=4,
                adapter='b c h w -> b (h w) c',
            ),
            schedule=LinearSchedule(),
            img_shape=(self.img_h, self.img_w),
            vocab_size=256,
        )

        imgs : Tensor = vdm(
            num_imgs,
            num_steps,
        )

        self.assertEqual(imgs.shape, (num_imgs, *self.img_shape))

    def test_compute_loss(self):

        imgs = torch.rand((self.batch_size, *self.img_shape))

        out_shape = (self.batch_size, self.chn_dim, self.img_h, self.img_w)

        vdm = VariationalDiffusion(
            backbone=UNet(
                net_dim=4,
                ctrl_dim=None,
                use_cond=False,
                use_attn=True,
                num_group=4,
                adapter='b c h w -> b (h w) c',
            ),
            schedule=LinearSchedule(),
            img_shape=(self.img_h, self.img_w),
            vocab_size=256,
        )

        loss, stat = vdm.compute_loss(
            imgs
        )

        print(stat)

        self.assertTrue(loss > 0)

    