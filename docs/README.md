# Code explanation

This directory explains how the library is built. Every document is self-contained and links to the source code and to related documents instead of repeating them.

## Map

| Topic | Document |
| --- | --- |
| GAN composition and the two-graph scheme | [gan.md](gan.md) |
| Numerical examples (step-by-step forward and backward) | [numeric](numeric) |
| Loss functions | [losses.md](losses.md) |
| Solvers: SGD, RMSProp, Adam | [solvers.md](solvers.md) |
| Gorgonia pitfalls learned the hard way | [pitfalls.md](pitfalls.md) |
| Conventions shared by recurrent layers | [layers/recurrent.md](layers/recurrent.md) |
| Linear | [layers/linear.md](layers/linear.md) |
| Convolutional | [layers/conv2d.md](layers/conv2d.md) |
| Maxpool | [layers/maxpool.md](layers/maxpool.md) |
| AvgPool | [layers/avgpool.md](layers/avgpool.md) |
| Flatten | [layers/flatten.md](layers/flatten.md) |
| Reshape | [layers/reshape.md](layers/reshape.md) |
| Dropout | [layers/dropout.md](layers/dropout.md) |
| Embedding | [layers/embedding.md](layers/embedding.md) |
| RNN | [layers/rnn.md](layers/rnn.md) |
| LSTM | [layers/lstm.md](layers/lstm.md) |
| GRU | [layers/gru.md](layers/gru.md) |

## Package layout

| File | Purpose |
| --- | --- |
| [layer.go](../layer.go) | `Layer` interface and helpers shared by layers (`addBias`, `singleInput`, `sliceGate`, `cloneLearnableTo`) |
| `layer_*.go` | One file per layer type implementing the interface |
| [network.go](../network.go) | `Network`: a sequence of layers with a single `Fwd` pass |
| [generator.go](../generator.go), [discriminator.go](../discriminator.go) | Thin wrappers around `Network` naming the two GAN roles |
| [gan.go](../gan.go) | `GAN`: composition of the two parts, see [gan.md](gan.md) |
| [loss.go](../loss.go) | Loss functions, see [losses.md](losses.md) |
| [activation.go](../activation.go) | Aliases to Gorgonia activation functions |
| [utils.go](../utils.go) | Dataset generation, plotting, hashing trick, padding |
| `*_test.go` | Tests, see the testing approach below |

## Core abstractions

Every layer implements four methods:

```go
type Layer interface {
	Fwd(inputs ...*gorgonia.Node) (*gorgonia.Node, error)
	Activate(input *gorgonia.Node) (*gorgonia.Node, error)
	Learnables() gorgonia.Nodes
	CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error)
}
```

`Fwd` builds the forward pass on the expression graph and returns the non-activated output node. Batch size is derived from shapes of the inputs, the first dimension is considered to be the batch one. `Activate` applies the activation function (layers without a natural activation just return the input). `Learnables` lists trainable nodes for solvers. `CloneTo` copies the layer structure onto another graph binding the same weight tensors, which is the backbone of the GAN composition (see [gan.md](gan.md)).

`Network` chains layers, `Generator` and `Discriminator` are named wrappers around it, `GAN` combines them across two graphs.

Note that all of this is graph construction time machinery: at training/inference time Gorgonia's tape machine executes the compiled graph and none of these interfaces participate.

## How to add a new layer

1. Create `layer_<name>.go` with a struct implementing the four interface methods. Keep only the fields your layer needs.
2. Respect the safety invariants of Gorgonia described in [pitfalls.md](pitfalls.md): unique names for value nodes, slice views only in unary operations, reshape before activation.
3. Add tests: forward pass against a reference computed with plain Go loops, a numerical gradient check (see `numericGradCheck` in [layer_gradients_test.go](../layer_gradients_test.go)), a `CloneTo` shared memory check.
4. Optionally add an example to `cmd/examples` and a document to `docs/layers`.

## Testing approach

- Forward passes of layers and losses are checked against references computed with plain Go loops and hand derived formulas, not against the library itself.
- Backward passes of recurrent layers are checked numerically: analytic gradients are compared with central finite differences.
- The shared memory assumption behind the GAN composition is guarded by dedicated tests in [gan_test.go](../gan_test.go), so any change of `gorgonia.WithValue` or solver semantics will be caught.

## A note on reproducibility

`rand.Seed(...)` in the examples fixes the generated datasets, but not the weights: Gorgonia's `GlorotN` [[1]](#references) initializer is seeded with the current time internally. Two runs of the same example produce the same data and different weights, so printed outputs vary between runs.

## References

```bibtex
% [1]
@inproceedings{glorot2010understanding,
    title={Understanding the difficulty of training deep feedforward neural networks},
    author={Xavier Glorot and Yoshua Bengio},
    booktitle={Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)},
    pages={249-256},
    year={2010},
    note={\url{https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf}}
}
```
