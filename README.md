# Variational Diffusion Models in Easy PyTorch

This repo is an _unofficial_ implementation of `Variational Diffusion Models` as introduced originally in [Kingma et al., (2021)](https://arxiv.org/abs/2107.00630) (revised in 2023). The authors provided an [official implementation](https://github.com/google-research/vdm) in JAX. Other PyTorch implementations exist (see [this nice example](https://github.com/addtt/variational-diffusion-models/tree/main)), so I developed this repo mainly for didactic purposes and as a gentle introduction to [`Bayesian Flow Networks`](https://arxiv.org/abs/2308.07037) that share similar principles.

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