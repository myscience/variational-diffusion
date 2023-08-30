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
    schedule=LinearSchedule(), # Linear schedule with learnable endpoints
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

We now support the learnable noise schedule (the $\gamma_\eta(t)$ network in the paper) via the `LearnableSchedule` module. This is implemented via a monotonic linear network (which uses the `MonotonicLinear` module) as described in *Constrained Monotonic Neural Networks* [Runje & Shankaranarayana, ICML (2023)](https://arxiv.org/abs/2205.11775). Moreover, we added preliminary support for optimizing the noise schedule to reduce the variance of the diffusion loss (as discussed in `Appendix I.2` of the main paper). This is achieved via the `reduce_variance` call, which re-uses the already-computed gradient needed for the VLB to reduce computational overhead. 

```python
import torch

from src.unet import UNet
from src.vdm import VariationalDiffusion
from src.schedule import LearnableSchedule

vdm = VariationalDiffusion(
    backbone=UNet(
        net_dim=4,
        ctrl_dim=None,
        use_cond=False,
        use_attn=True,
        num_group=4,
        adapter='b c h w -> b (h w) c',
    ),
    schedule=LearnableSchedule(
      hid_dim=[50, 50],
      gate_func='relu',
    ), # Linear schedule with learnable endpoints
    img_shape=(32, 32),
    vocab_size=256,
)

# Get some fake imgs for testing
imgs = torch.randn(16, 3, 32, 32)

# Initialize the optimizer of choice
optim = torch.optim.AdamW(vdm.paramters(), lr=1e-3)
optim.zero_grad()

# First we compute the VLB loss
loss, stats = vdm.compute_loss(imgs)

# Then we call .backward() to populate the gradients
# NOTE: We need to retain the graph to access the
#       gradients, otherwise they are freed
loss.backward(retain_graph=True)

# Finally we update the noise-schedule gradients to
# support lower variance (faster training)
vdm.reduce_variance(*stats['var_args'])

# Finally we update the model parameters
optim.step()

# We can manually delete both loss and stat to put
# the grad graph out-of-scope so it gets freed
def loss, stats
```

# Roadmap

- [x] Put all the essential pieces together: UNet, VDM, a noise schedule.
- [x] Add fully learnable schedule (monotonic neural network). Implement gradient trick described in Appendix I.2
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