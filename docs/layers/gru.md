# GRU layer

Source: [layer_gru.go](../../layer_gru.go)

Gated recurrent unit [[1]](#references) in the original formulation of Cho et al.: the reset gate is applied to the previous hidden state before the projection. All three gates are packed into single weight matrices, columns in order: reset gate $r$, update gate $z$, candidate $n$. For every time step $t$:

$$\begin{aligned}
r_t &= \sigma\big((x_t W + b)_r + (h_{t-1} U)_r\big) \\
z_t &= \sigma\big((x_t W + b)_z + (h_{t-1} U)_z\big) \\
n_t &= \tanh\big((x_t W + b)_n + ((r_t \odot h_{t-1}) U)_n\big) \\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1}
\end{aligned} \tag{1}$$

Shared conventions are described in [recurrent.md](recurrent.md). Compared to [LSTM](lstm.md) the GRU has no separate cell state and one gate less, with comparable quality on many tasks [[2]](#references).

## Shapes

| Tensor | Shape |
| --- | --- |
| `InputWeightNode` $W$ | $[F, 3H]$ |
| `HiddenWeightNode` $U$ | $[H, 3H]$ |
| `BiasNode` $b$ (optional) | $[1, 3H]$ |

## Backward pass

Split the hidden weights by gate columns: $U = [\, U_r \;\; U_z \;\; U_n \,]$. Let $\gamma_t$ be the direct gradient on $h_t$. Going backwards in time, at step $t$ with the total hidden gradient $\delta^h_t$:

$$\begin{aligned}
dz_t &= \delta^h_t \odot (h_{t-1} - n_t) \odot z_t (1 - z_t) \\
dn_t &= \delta^h_t \odot (1 - z_t) \odot (1 - n_t^2) \\
dr_t &= (dn_t U_n^{\top}) \odot h_{t-1} \odot r_t (1 - r_t)
\end{aligned} \tag{2}$$

and the recurrent part of the previous step gradient collects four paths:

$$\delta^h_{t-1} = \gamma_{t-1} + \underbrace{\delta^h_t \odot z_t}_{\text{identity path}} + \underbrace{(dn_t U_n^{\top}) \odot r_t}_{\text{candidate}} + \underbrace{dr_t U_r^{\top} + dz_t U_z^{\top}}_{\text{gates}} \tag{3}$$

$$\frac{\partial L}{\partial W} = \sum_t x_t^{\top} [\, dr_t \;\; dz_t \;\; dn_t \,] \qquad \frac{\partial L}{\partial b} = \sum_t \sum_{batch} [\, dr_t \;\; dz_t \;\; dn_t \,] \tag{4}$$

$$\frac{\partial L}{\partial U_r} = \sum_t h_{t-1}^{\top} dr_t \qquad \frac{\partial L}{\partial U_z} = \sum_t h_{t-1}^{\top} dz_t \qquad \frac{\partial L}{\partial U_n} = \sum_t (r_t \odot h_{t-1})^{\top} dn_t \tag{5}$$

The identity path $\delta^h_t \odot z_t$ of (3) plays the same role as the forget gate of [LSTM](lstm.md): with update gates close to one the gradient flows back through time without touching weight matrices or activation derivatives.

## Usage

```go
layer := &gan.GRULayer{
	InputWeightNode:  w,
	HiddenWeightNode: u,
	BiasNode:         b,
	HiddenSize:       32,
}
```

`Activation` replaces $\tanh$ (candidate), `RecurrentActivation` replaces $\sigma$ (reset and update gates).

## Implementation notes

The hidden state update is computed as $n_t + z_t \odot (h_{t-1} - n_t)$, which is algebraically the same and avoids a ones tensor. Per-step outputs are detached from hidden state buffers through a zero-valued anchor node: this layer is the reason pitfall 2 of [pitfalls.md](../pitfalls.md) was discovered. Forward pass is verified against a reference computed with plain Go loops, gradients numerically ([layer_gru_test.go](../../layer_gru_test.go), [layer_gradients_test.go](../../layer_gradients_test.go)). Complete usage: [cmd/examples/train_gru](../../cmd/examples/train_gru/main.go).

## References

```bibtex
% [1]
@inproceedings{cho2014learning,
    title={Learning phrase representations using RNN encoder-decoder for statistical machine translation},
    author={Kyunghyun Cho and Bart van Merrienboer and Caglar Gulcehre and Dzmitry Bahdanau and Fethi Bougares and Holger Schwenk and Yoshua Bengio},
    booktitle={Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    pages={1724-1734},
    year={2014},
    note={\url{https://arxiv.org/abs/1406.1078}}
}
% [2]
@inproceedings{chung2014empirical,
    title={Empirical evaluation of gated recurrent neural networks on sequence modeling},
    author={Junyoung Chung and Caglar Gulcehre and Kyunghyun Cho and Yoshua Bengio},
    booktitle={NIPS 2014 Workshop on Deep Learning},
    year={2014},
    note={\url{https://arxiv.org/abs/1412.3555}}
}
```
