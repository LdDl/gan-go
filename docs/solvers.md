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

## Numerical example

One parameter vector, the same three gradients fed to all three solvers, so the trajectories are directly comparable. Generated and verified step by step against the actual Gorgonia solvers by [cmd/docsgen](../cmd/docsgen).

$$\theta^{(0)} = (1,\; -2) \qquad g_1 = (2,\; -4) \qquad g_2 = (1,\; 2) \qquad g_3 = (-3,\; 1)$$

Common settings: $\eta = 0.1$, $B = 1$. For RMSProp $\rho = 0.9$ is set explicitly (the value of the original lecture), for Adam $\beta_1 = 0.9$, $\beta_2 = 0.999$, both with $\varepsilon = 10^{-8}$.

### Vanilla SGD steps

First component of step 1: $\theta_0 = 1 - 0.1 \cdot 2 = 0.8$. Every step subtracts the plain scaled gradient:

| $t$ | $g_t$ | $\theta^{(t)}$ |
| --- | --- | --- |
| 0 | | $(1,\; -2)$ |
| 1 | $(2,\; -4)$ | $(0.8,\; -1.6)$ |
| 2 | $(1,\; 2)$ | $(0.7,\; -1.8)$ |
| 3 | $(-3,\; 1)$ | $(1,\; -1.9)$ |

Note the third step: the gradient flipped sign and the parameter walks back. The step size depends only on the gradient magnitude.

### RMSProp steps

First component of step 1: $r = 0.9 \cdot 0 + 0.1 \cdot 2^2 = 0.4$, then $\theta_0 = 1 - 0.1 \cdot 2 / \sqrt{0.4} \approx 0.6838$:

| $t$ | $g_t$ | $r_t$ | $\theta^{(t)}$ |
| --- | --- | --- | --- |
| 0 | | $(0,\; 0)$ | $(1,\; -2)$ |
| 1 | $(2,\; -4)$ | $(0.4,\; 1.6)$ | $(0.6838,\; -1.684)$ |
| 2 | $(1,\; 2)$ | $(0.46,\; 1.84)$ | $(0.5363,\; -1.831)$ |
| 3 | $(-3,\; 1)$ | $(1.314,\; 1.756)$ | $(0.798,\; -1.907)$ |

The division by $\sqrt{r_t}$ normalizes the step: components with large gradients (the second one) move no faster than components with small ones.

### Adam steps

First component of step 1: $m = 0.1 \cdot 2 = 0.2$, $v = 0.001 \cdot 4 = 0.004$, bias corrections $\hat{m} = 0.2/0.1 = 2$, $\hat{v} = 0.004/0.001 = 4$, so $\theta_0 = 1 - 0.1 \cdot 2/\sqrt{4} = 0.9$. The corrections exactly undo the zero initialization at $t = 1$:

| $t$ | $g_t$ | $m_t$ | $v_t$ | $\theta^{(t)}$ |
| --- | --- | --- | --- | --- |
| 0 | | $(0,\; 0)$ | $(0,\; 0)$ | $(1,\; -2)$ |
| 1 | $(2,\; -4)$ | $(0.2,\; -0.4)$ | $(0.004,\; 0.016)$ | $(0.9,\; -1.9)$ |
| 2 | $(1,\; 2)$ | $(0.28,\; -0.16)$ | $(0.004996,\; 0.01998)$ | $(0.8068,\; -1.873)$ |
| 3 | $(-3,\; 1)$ | $(-0.048,\; -0.044)$ | $(0.01399,\; 0.02096)$ | $(0.815,\; -1.867)$ |

Adam steps stay close to $\eta$ in magnitude regardless of the raw gradient scale, and the momentum $m_t$ smooths the sign flip of $g_3$: compare the third step with the vanilla one.

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
