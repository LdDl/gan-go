# Linear layer

Source: [layer_linear.go](../../layer_linear.go)

Fully connected (dense) layer: the affine transformation at the core of the perceptron [[1]](#references) and of multilayer networks trained with backpropagation [[2]](#references):

$$y = x W^{\top} + b \tag{1}$$

## Shapes

| Tensor | Shape |
| --- | --- |
| Input $x$ | $[B, F_{in}]$ |
| `WeightNode` $W$ | $[F_{out}, F_{in}]$ |
| `BiasNode` $b$ (optional) | $[1, F_{out}]$ |
| Output $y$ | $[B, F_{out}]$ |

Note that $W$ is stored as $[F_{out}, F_{in}]$ and transposed inside `Fwd`, so rows of the weight matrix correspond to output neurons.

## Backward pass

Let $\delta = \partial L / \partial y$ of shape $[B, F_{out}]$ be the gradient coming from the next layer. Then

$$\frac{\partial L}{\partial W} = \delta^{\top} x \qquad \frac{\partial L}{\partial b} = \sum_{batch} \delta \qquad \frac{\partial L}{\partial x} = \delta \, W \tag{2}$$

The batch dimension is contracted away in the weight gradient and summed in the bias gradient, which is why solvers of the examples are configured with `gorgonia.WithBatchSize(...)` to average the update. Gorgonia derives these expressions automatically from the graph.

## Usage

```go
layer := &gan.LinearLayer{
	WeightNode: w,
	BiasNode:   b,
	Activation: gan.Rectify,
}
```

## Implementation notes

Plain matrix multiplication handles a 2D input of any batch size at once. For inputs of higher dimensions `BatchedMatMul` is applied. The bias is added with broadcasting along the batch dimension when $B > 1$ (the shared `addBias` helper in [layer.go](../../layer.go)).

## References

```bibtex
% [1]
@article{rosenblatt1958perceptron,
    title={The perceptron: a probabilistic model for information storage and organization in the brain},
    author={Frank Rosenblatt},
    journal={Psychological Review},
    volume={65},
    number={6},
    pages={386-408},
    year={1958},
    note={\url{https://homepages.math.uic.edu/~lreyzin/papers/rosenblatt58.pdf}}
}
% [2]
@article{rumelhart1986learning,
    title={Learning representations by back-propagating errors},
    author={David E. Rumelhart and Geoffrey E. Hinton and Ronald J. Williams},
    journal={Nature},
    volume={323},
    pages={533-536},
    year={1986},
    note={\url{https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf}}
}
```
