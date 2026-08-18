# Loss functions

All losses live in [loss.go](../loss.go). Every function takes prediction node $a$, target node $b$ and an optional reduction (`LossReductionMean` by default, `LossReductionSum` available).

## MSE

$$L(a, b) = \frac{1}{n}\sum_{i=1}^{n} (a_i - b_i)^2 \tag{1}$$

Used by most GAN examples of this repository, which corresponds to the least squares GAN objective (see [gan.md](gan.md)).

## L1

$$L(a, b) = \frac{1}{n}\sum_{i=1}^{n} |a_i - b_i| \tag{2}$$

## Cross entropy

$$L(a, b) = -\frac{1}{n}\sum_{i=1}^{n} b_i \log(a_i + \varepsilon) \tag{3}$$

Standard loss for classification and language modeling, used by the recurrent examples together with a softmax output layer. See [[1]](#references) for the information-theoretic definition (section 3.13), the connection to maximum likelihood (section 5.5) and its use as a cost function of neural networks (section 6.2).

## Binary cross entropy

$$L(a, b) = -\frac{1}{n}\sum_{i=1}^{n} \left[ b_i \log(a_i + \varepsilon) + (1 - b_i) \log(1 - a_i + \varepsilon) \right] \tag{4}$$

The two-class variant of cross entropy, used with a sigmoid output. This is the loss of the classic GAN objective.

## Pseudo-Huber

$$L(a, b) = \frac{1}{n}\sum_{i=1}^{n} \delta^2 \left( \sqrt{1 + \left(\frac{a_i - b_i}{\delta}\right)^2} - 1 \right) \tag{5}$$

A smooth approximation of the Huber loss [[2]](#references), known as the pseudo-Huber or Charbonnier loss [[3]](#references): quadratic for small residuals, asymptotically linear for large ones. The `delta` argument must match the dtype of the nodes (`float64` value for `Float64` graphs and so on).

## Gradients

Gorgonia derives gradients automatically by symbolic differentiation of the graph. Formulas (6), (7) and (8) below are the derivatives of the losses (1) to (5) for the mean reduction ($x_i = a_i - b_i$ where convenient):

$$\frac{\partial L_{MSE}}{\partial a_i} = \frac{2 (a_i - b_i)}{n} \qquad \frac{\partial L_{L1}}{\partial a_i} = \frac{\operatorname{sign}(a_i - b_i)}{n} \tag{6}$$

$$\frac{\partial L_{CE}}{\partial a_i} = -\frac{b_i}{n \, (a_i + \varepsilon)} \qquad \frac{\partial L_{BCE}}{\partial a_i} = \frac{1}{n}\left( \frac{1 - b_i}{1 - a_i + \varepsilon} - \frac{b_i}{a_i + \varepsilon} \right) \tag{7}$$

$$\frac{\partial L_{PH}}{\partial a_i} = \frac{x_i}{n \sqrt{1 + (x_i/\delta)^2}} \tag{8}$$

Two observations worth teaching:

- The pseudo-Huber gradient behaves as $x_i/n$ (linear, like MSE) for $|x_i| \ll \delta$ and saturates at $\pm\delta/n$ for $|x_i| \gg \delta$, which is exactly the robustness to outliers the loss is used for.
- Without the epsilon shift the cross entropy gradients contain $1/a_i$, which is unbounded. The shift caps their magnitude at $1/\varepsilon$, see below.

## The epsilon shift

Both cross entropy variants compute $\log(A + \varepsilon)$ instead of $\log(A)$, with $\varepsilon = 10^{-7}$ for `float32` and $10^{-12}$ for `float64`. A saturated activation (output exactly 0.0 or 1.0) would otherwise produce an infinite loss and NaN gradients, and a single NaN irreversibly poisons the state of solvers like Adam or RMSProp. With the shift the loss is capped at $-\log \varepsilon$ and the gradient magnitude at $1/\varepsilon$. For non saturated outputs the difference is below $10^{-12}$. Mainstream frameworks guard the same way: Keras clips probabilities to $[\varepsilon, 1 - \varepsilon]$, PyTorch clamps log outputs of `BCELoss` at $-100$.

Nodes holding the epsilon value are named uniquely per call site. The reason is a Gorgonia pitfall described in [pitfalls.md](pitfalls.md): value nodes are deduplicated by type, shape and name, the value itself is not part of the hash.

## References

```bibtex
% [1] (sections 3.13, 5.5 and 6.2)
@book{goodfellow2016deep,
    title={Deep Learning},
    author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
    publisher={MIT Press},
    year={2016},
    note={\url{https://www.deeplearningbook.org}}
}
% [2]
@article{huber1964robust,
    title={Robust estimation of a location parameter},
    author={Peter J. Huber},
    journal={The Annals of Mathematical Statistics},
    volume={35},
    number={1},
    pages={73-101},
    year={1964},
    note={\url{https://doi.org/10.1214/aoms/1177703732}}
}
% [3]
@inproceedings{barron2019general,
    title={A general and adaptive robust loss function},
    author={Jonathan T. Barron},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={4331-4339},
    year={2019},
    note={\url{https://arxiv.org/abs/1701.03077}}
}
```
