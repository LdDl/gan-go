# Conventions shared by recurrent layers

[RNN](rnn.md), [LSTM](lstm.md) and [GRU](gru.md) follow the same set of conventions.

## Shapes

Input is a sequence tensor of shape $[T, F]$ or $[T, B, F]$, where $T$ is the sequence length, $B$ is the batch size and $F$ is the number of input features. A 2D input is treated as a batch of one. Output holds hidden states of every time step: $[T, H]$ or $[T, B, H]$ respectively, where $H$ is the hidden size.

Weights pack all gates into single matrices: the input weights have shape $[F, kH]$ and the hidden weights $[H, kH]$, where $k$ is the number of gates (1 for RNN, 3 for GRU, 4 for LSTM). The optional bias has shape $[1, kH]$. Gates are extracted as column slices, the packing order is documented in each layer.

## Initial state

Zero-valued initial states are created automatically with names unique per graph (see pitfall 1 in [pitfalls.md](../pitfalls.md)). Custom initial states could be provided in two ways: via the `Initial*Node` fields of the layer or as extra input nodes of the `Fwd` call (second node for the hidden state, third for the LSTM cell state). Expected shape is $[B, H]$.

## Final state

`FinalHidden()` (and `FinalCell()` for LSTM) return nodes of the last time step states after `Fwd` was called. They are useful for sequence-to-one architectures and for chaining.

## Activation

The `Activate` method of recurrent layers returns the input unchanged: activation functions are applied inside the cell and are configurable via the `Activation` and `RecurrentActivation` fields (Tanh and Sigmoid by default).

## Graph unrolling

Gorgonia builds static graphs, so the recurrence is unrolled at graph construction time: the time loop inside `Fwd` creates $T$ copies of the cell subgraph. Gradients flow through the whole unrolled chain, which is exactly backpropagation through time [[1]](#references).

## Backpropagation through time

Since every hidden state contributes to the output, the loss gradient with respect to $h_t$ has two sources: the direct one $\gamma_t$ (through the output at step $t$) and the recurrent one propagated from step $t+1$. Weight gradients accumulate contributions of all steps:

$$\frac{\partial L}{\partial \theta} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial \theta} \tag{1}$$

The per-layer documents spell out the recursions: [rnn.md](rnn.md) (including why gradients of the plain recurrence vanish or explode), [lstm.md](lstm.md) (the additive cell path), [gru.md](gru.md) (the update gate as an interpolation highway). All of these formulas are what Gorgonia derives automatically from the unrolled graph, and the numerical gradient checks in [layer_gradients_test.go](../../layer_gradients_test.go) verify that the analytic gradients match central finite differences.

## Safety invariants

The implementation follows the rules described in [pitfalls.md](../pitfalls.md): slice views are passed to unary operations only, gate extraction reshapes before activating, per-step outputs are detached from hidden state buffers where needed.

## References

```bibtex
% [1]
@article{werbos1990backpropagation,
    title={Backpropagation through time: what it does and how to do it},
    author={Paul J. Werbos},
    journal={Proceedings of the IEEE},
    volume={78},
    number={10},
    pages={1550-1560},
    year={1990},
    note={\url{http://www.werbos.com/Neural/BTT.pdf}}
}
```
