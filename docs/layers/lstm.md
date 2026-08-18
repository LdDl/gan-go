# LSTM layer

Source: [layer_lstm.go](../../layer_lstm.go)

Long short-term memory layer [[1]](#references) with forget gate [[2]](#references). All four gates are packed into single weight matrices, columns in order: input gate $i$, forget gate $f$, cell candidate $g$, output gate $o$. For every time step $t$:

$$\begin{aligned}
i_t &= \sigma\big((x_t W + b)_i + (h_{t-1} U)_i\big) \\
f_t &= \sigma\big((x_t W + b)_f + (h_{t-1} U)_f\big) \\
g_t &= \tanh\big((x_t W + b)_g + (h_{t-1} U)_g\big) \\
o_t &= \sigma\big((x_t W + b)_o + (h_{t-1} U)_o\big) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned} \tag{1}$$

Shared conventions are described in [recurrent.md](recurrent.md). The cell state gives the layer an additive memory path, mitigating the vanishing gradients of the plain [RNN](rnn.md).

## Shapes

| Tensor | Shape |
| --- | --- |
| `InputWeightNode` $W$ | $[F, 4H]$ |
| `HiddenWeightNode` $U$ | $[H, 4H]$ |
| `BiasNode` $b$ (optional) | $[1, 4H]$ |

## Backward pass

Let $\gamma_t$ be the direct gradient on $h_t$ and $\delta^{gates}_{T+1} = 0$, $\delta^c_{T+1} = 0$. Going backwards in time, the hidden and cell gradients are

$$\delta^h_t = \gamma_t + \delta^{gates}_{t+1} U^{\top} \qquad \delta^c_t = \delta^h_t \odot o_t \odot (1 - \tanh^2(c_t)) + \delta^c_{t+1} \odot f_{t+1} \tag{2}$$

and the pre-activation gate gradients (packed in the same column order as the forward pass)

$$\begin{aligned}
di_t &= \delta^c_t \odot g_t \odot i_t (1 - i_t) \\
df_t &= \delta^c_t \odot c_{t-1} \odot f_t (1 - f_t) \\
dg_t &= \delta^c_t \odot i_t \odot (1 - g_t^2) \\
do_t &= \delta^h_t \odot \tanh(c_t) \odot o_t (1 - o_t) \\
\delta^{gates}_t &= [\, di_t \;\; df_t \;\; dg_t \;\; do_t \,]
\end{aligned} \tag{3}$$

$$\frac{\partial L}{\partial W} = \sum_t x_t^{\top} \delta^{gates}_t \qquad \frac{\partial L}{\partial U} = \sum_t h_{t-1}^{\top} \delta^{gates}_t \qquad \frac{\partial L}{\partial b} = \sum_t \sum_{batch} \delta^{gates}_t \tag{4}$$

The key structural property is visible in the $\delta^c$ recursion of (2): the cell-to-cell path multiplies the gradient by $f_{t+1}$ only. No weight matrix and no activation derivative participate, so with forget gates close to one the error flows back through time almost unchanged (the constant error carousel of [[1]](#references)). Compare with the $\operatorname{diag}(1-h^2) U^{\top}$ factor of the plain [RNN](rnn.md).

## Usage

```go
layer := &gan.LSTMLayer{
	InputWeightNode:  w,
	HiddenWeightNode: u,
	BiasNode:         b,
	HiddenSize:       32,
}
```

`Activation` replaces $\tanh$ (cell candidate and cell output), `RecurrentActivation` replaces $\sigma$ (gates). `FinalCell()` exposes the cell state of the last step in addition to `FinalHidden()`.

## Implementation notes

Gates are extracted with the `sliceGate` helper respecting the invariants of [pitfalls.md](../pitfalls.md). Forward pass is verified against a reference computed with plain Go loops, gradients numerically ([layer_lstm_test.go](../../layer_lstm_test.go), [layer_gradients_test.go](../../layer_gradients_test.go)). Complete usage: [cmd/examples/train_lstm](../../cmd/examples/train_lstm/main.go).

## References

```bibtex
% [1]
@article{hochreiter1997long,
    title={Long short-term memory},
    author={Sepp Hochreiter and J{\"u}rgen Schmidhuber},
    journal={Neural Computation},
    volume={9},
    number={8},
    pages={1735-1780},
    year={1997},
    note={\url{https://www.bioinf.jku.at/publications/older/2604.pdf}}
}
% [2]
@article{gers2000learning,
    title={Learning to forget: continual prediction with LSTM},
    author={Felix A. Gers and J{\"u}rgen Schmidhuber and Fred Cummins},
    journal={Neural Computation},
    volume={12},
    number={10},
    pages={2451-2471},
    year={2000},
    note={\url{https://direct.mit.edu/neco/article-pdf/12/10/2451/814643/089976600300015015.pdf}}
}
```
