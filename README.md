# Variational Diffusion Models in Easy PyTorch

This repo is an _unofficial_ implementation of `Variational Diffusion Models` as introduced originally in [Kingma et al., (2021)](https://arxiv.org/abs/2107.00630) (revised in 2023). The authors provided an [official implementation](https://github.com/google-research/vdm) in JAX. Other PyTorch implementations exist (see [this nice example](https://github.com/addtt/variational-diffusion-models/tree/main)), so I developed this repo mainly for didactic purposes and as a gentle introduction to [`Bayesian Flow Networks`](https://arxiv.org/abs/2308.07037) that share similar principles.

## Usage

```python
import torch

from src.unet import UNet
from src.vdm import VariationalDiffusion
from src.schedule import LinearSchedule

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
    img_shape=(32, 32),
    vocab_size=256,
)

# Get some fake imgs for testing
imgs = torch.randn(16, 3, 32, 32)

# Compute the VDM loss, which is a combination of
# diffusion + latent + reconstruction loss
loss, stats = vdm.compute_loss(imgs)

# Once the model is trained, we can sample from the learnt
# inverse diffusion process by simply doing
num_imgs = 4
num_step = 100

samples = vdm(num_imgs, num_step)
```

# Roadmap

- [x] Put all the essential pieces together: UNet, VDM, a noise schedule.
- [ ] Add fully learnable schedule (monotonic neural network). Implement gradient trick described in Appendix I.2
- [ ] Add functioning training script (Lightning).
- [ ] Show some results.

## Citations

```bibtex
@article{kingma2021variational,
  title={Variational diffusion models},
  author={Kingma, Diederik and Salimans, Tim and Poole, Ben and Ho, Jonathan},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={21696--21707},
  year={2021}
}
```

```bibtex
@inproceedings{runje2023constrained,
  title={Constrained monotonic neural networks},
  author={Runje, Davor and Shankaranarayana, Sharath M},
  booktitle={International Conference on Machine Learning},
  pages={29338--29353},
  year={2023},
  organization={PMLR}
}
```