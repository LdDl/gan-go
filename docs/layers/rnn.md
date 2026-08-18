# RNN layer

Source: [layer_rnn.go](../../layer_rnn.go)

Vanilla (Elman) recurrent layer [[1]](#references). For every time step $t$:

$$h_t = \tanh(x_t W + h_{t-1} U + b) \tag{1}$$

Shared conventions (shapes, initial and final states, unrolling) are described in [recurrent.md](recurrent.md). For gated variants mitigating the vanishing gradient problem of this architecture see [lstm.md](lstm.md) and [gru.md](gru.md).

## Shapes

| Tensor | Shape |
| --- | --- |
| `InputWeightNode` $W$ | $[F, H]$ |
| `HiddenWeightNode` $U$ | $[H, H]$ |
| `BiasNode` $b$ (optional) | $[1, H]$ |

## Backward pass

Write the step as $h_t = \tanh(a_t)$ with $a_t = x_t W + h_{t-1} U + b$. Let $\gamma_t$ be the direct gradient on $h_t$ (through the output at step $t$) and $\delta^a_{T+1} = 0$. Going backwards in time:

$$\delta^h_t = \gamma_t + \delta^a_{t+1} U^{\top} \qquad \delta^a_t = \delta^h_t \odot (1 - h_t \odot h_t) \tag{2}$$

$$\frac{\partial L}{\partial W} = \sum_{t} x_t^{\top} \delta^a_t \qquad \frac{\partial L}{\partial U} = \sum_{t} h_{t-1}^{\top} \delta^a_t \qquad \frac{\partial L}{\partial b} = \sum_{t} \sum_{batch} \delta^a_t \tag{3}$$

The recurrent factor applied per step in (2) is $\operatorname{diag}(1 - h_t^2)\, U^{\top}$. Over $k$ steps the gradient is multiplied by $k$ such factors, so it shrinks exponentially when their norms are below one and explodes otherwise. This is the vanishing/exploding gradient problem of plain recurrences [[2]](#references), the motivation behind [LSTM](lstm.md) and [GRU](gru.md).

## Usage

```go
layer := &gan.RNNLayer{
	InputWeightNode:  w,
	HiddenWeightNode: u,
	BiasNode:         b,
	HiddenSize:       32,
}
```

`Activation` replaces $\tanh$ when set.

## Implementation notes

Forward pass is verified against a reference computed with plain Go loops, gradients through the unrolled recurrence are verified numerically ([layer_rnn_test.go](../../layer_rnn_test.go), [layer_gradients_test.go](../../layer_gradients_test.go)). Complete usage: [cmd/examples/train_rnn](../../cmd/examples/train_rnn/main.go).

## References

```bibtex
% [1]
@article{elman1990finding,
    title={Finding structure in time},
    author={Jeffrey L. Elman},
    journal={Cognitive Science},
    volume={14},
    number={2},
    pages={179-211},
    year={1990},
    note={\url{https://gwern.net/doc/ai/nn/rnn/1990-elman.pdf}}
}
% [2]
@article{bengio1994learning,
    title={Learning long-term dependencies with gradient descent is difficult},
    author={Yoshua Bengio and Patrice Simard and Paolo Frasconi},
    journal={IEEE Transactions on Neural Networks},
    volume={5},
    number={2},
    pages={157-166},
    year={1994},
    note={\url{https://www.iro.umontreal.ca/~lisa/pointeurs/ieeetrnn94.pdf}}
}
```
