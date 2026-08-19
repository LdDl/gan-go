# Solvers

Solvers consume the gradients derived in [layers](layers) documents (see the [linear numerical example](numeric/linear.md) for gradients on concrete numbers) and update the learnable parameters in place. The in-place property matters beyond performance: the whole GAN composition relies on it, see [gan.md](gan.md).

All formulas below are written for a single parameter tensor $\theta$ with gradient $g_t$ at step $t$, and match the actual implementation of Gorgonia v0.9 (defaults included). $B$ is the value of `gorgonia.WithBatchSize`.

## Vanilla SGD

`NewVanillaSolver`. Plain gradient descent, the step of the [linear numerical example](numeric/linear.md):

$$\theta \leftarrow \theta - \frac{\eta}{B} \, g_t \tag{1}$$

One learning rate for every parameter. Simple and predictable, but the same step scale is applied to rarely and frequently updated parameters alike, which is what the adaptive methods below fix.

## RMSProp

`NewRMSPropSolver`. Proposed in a lecture by Tieleman and Hinton [[1]](#references): keep a running average of the squared gradient per parameter and divide the step by its root, so parameters with consistently large gradients take smaller steps:

$$r_t = \rho \, r_{t-1} + (1 - \rho) \, g_t^2 \tag{2}$$

$$\theta \leftarrow \theta - \eta \, \frac{g_t}{\sqrt{r_t + \varepsilon}} \tag{3}$$

Gorgonia defaults: $\eta = 0.001$, $\rho = 0.999$, $\varepsilon = 10^{-8}$. Two implementation details worth knowing:

- $\varepsilon$ sits INSIDE the square root in Gorgonia, some textbook formulations place it outside. The difference is negligible in practice but visible in exact reproductions.
- `WithBatchSize` is not supported by the RMSProp solver of Gorgonia v0.9 and is silently ignored (the option applies to Adam, Vanilla, Momentum and AdamW only). The GAN examples of this repository pass it to RMSProp for uniformity, it has no effect there.

Note the default $\rho = 0.999$ of Gorgonia differs from the $\rho = 0.9$ suggested in the original lecture: the running average is much slower.

Used by the GAN examples: [parabola](../cmd/examples/parabola/main.go), [sin](../cmd/examples/sin/main.go), [generate_symbol](../cmd/examples/generate_symbol/main.go), [generate_smiley_face](../cmd/examples/generate_smiley_face/main.go), [train_cnn](../cmd/examples/train_cnn/main.go).

## Adam

`NewAdamSolver`. Adaptive moment estimation [[2]](#references): RMSProp-style scaling plus momentum, both with bias correction for the zero initialization of the accumulators:

$$m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, \frac{g_t}{B} \qquad v_t = \beta_2 \, v_{t-1} + (1 - \beta_2) \, \frac{g_t^2}{B^2} \tag{4}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \tag{5}$$

$$\theta \leftarrow \theta - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} \tag{6}$$

Gorgonia defaults: $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, $B = 1$. The bias corrections of (5) matter early in training: $m$ and $v$ start at zero, so without the corrections the first steps would be strongly underestimated.

Used by the sequence examples: [train_embedding](../cmd/examples/train_embedding/main.go), [train_rnn](../cmd/examples/train_rnn/main.go), [train_lstm](../cmd/examples/train_lstm/main.go), [train_gru](../cmd/examples/train_gru/main.go).

## See also

- [Numerical example](numeric/solvers.md): the three solvers traced step by step over the same gradients, with the accumulators of every step.

## References

```bibtex
% [1]
@misc{tieleman2012rmsprop,
    title={Lecture 6.5-rmsprop: divide the gradient by a running average of its recent magnitude},
    author={Tijmen Tieleman and Geoffrey Hinton},
    howpublished={COURSERA: Neural Networks for Machine Learning, lecture slides},
    year={2012},
    note={\url{https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf}}
}
% [2]
@inproceedings{kingma2015adam,
    title={Adam: a method for stochastic optimization},
    author={Diederik P. Kingma and Jimmy Ba},
    booktitle={Proceedings of the 3rd International Conference on Learning Representations (ICLR)},
    year={2015},
    note={\url{https://arxiv.org/abs/1412.6980}}
}
```
