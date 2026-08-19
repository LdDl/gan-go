# Dropout layer

Source: [layer_dropout.go](../../layer_dropout.go)

Regularization layer [[1]](#references): during training every element of the input is dropped (zeroed) with probability $p$, and the survivors are scaled to keep the expectation unchanged (the inverted dropout scheme):

$$y_i = \frac{m_i \cdot x_i}{1 - p}, \qquad m_i \sim \text{Bernoulli}(1 - p) \tag{1}$$

so $\mathbb{E}[y_i] = x_i$.

The layer has no learnable parameters and no activation, `Activate` returns the input unchanged.

## Backward pass

The same mask and scaling as in the forward pass:

$$\frac{\partial L}{\partial x_i} = \frac{m_i}{1 - p} \cdot \delta_i \tag{2}$$

Gradients of dropped elements are zeroed, gradients of survivors are scaled by $1/(1-p)$.

## Usage

```go
layer := &gan.DropoutLayer{
	Probability: 0.3,
}
```

`Probability` outside of $[0, 1]$ is rejected with an error.

## Implementation notes

The computation is delegated to `gorgonia.Dropout`, which implements exactly the inverted scheme above and uses its own time-seeded random generator (one of the reasons example outputs vary between runs, see the reproducibility note in [README.md](../README.md)).

## References

```bibtex
% [1]
@article{srivastava2014dropout,
    title={Dropout: a simple way to prevent neural networks from overfitting},
    author={Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov},
    journal={Journal of Machine Learning Research},
    volume={15},
    number={1},
    pages={1929-1958},
    year={2014},
    note={\url{https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf}}
}
```
