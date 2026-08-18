# Embedding layer

Source: [layer_embedding.go](../../layer_embedding.go)

Lookup table [[1]](#references) mapping integer indices to dense vectors: output for index $t$ is the row $E_t$ of the weight matrix $E$ of shape $[V, d]$, where $V$ is the vocabulary size and $d$ is the embedding dimension.

$$y_t = E_{x_t} \tag{1}$$

## Shapes

| Tensor | Shape |
| --- | --- |
| Input | integer vector $[T]$ |
| `WeightNode` $E$ | $[V, d]$ |
| Output | $[T, d]$ |

The input node must be of Gorgonia type `Vector int`.

## Backward pass

The lookup is a row selection, so the gradient scatters back into the selected rows and accumulates over repeated indices:

$$\frac{\partial L}{\partial E_v} = \sum_{t:\; x_t = v} \delta_t \tag{2}$$

Rows of words absent from the batch receive zero gradient and are not updated at that step. This sparsity is why embedding layers train efficiently even for large vocabularies.

## Usage

```go
layer := &gan.EmbeddingLayer{
	WeightNode:    e,
	EmbeddingSize: 16,
}
```

## Implementation notes

Implemented via `gorgonia.ByIndices` over rows of the weight matrix. The whole matrix is learnable, so embeddings are trained together with the rest of the network. Text preprocessing helpers live in [utils.go](../../utils.go): `HashingTrick` [[2]](#references) maps words to indices of a fixed vocabulary and `PaddingInt64Slice` pads sequences to a fixed length. Complete usage: [cmd/examples/train_embedding](../../cmd/examples/train_embedding/main.go) and the recurrent examples ([rnn](rnn.md), [lstm](lstm.md), [gru](gru.md)).

## References

```bibtex
% [1]
@article{bengio2003neural,
    title={A neural probabilistic language model},
    author={Yoshua Bengio and R{\'e}jean Ducharme and Pascal Vincent and Christian Janvin},
    journal={Journal of Machine Learning Research},
    volume={3},
    pages={1137-1155},
    year={2003},
    note={\url{https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf}}
}
% [2]
@inproceedings{weinberger2009feature,
    title={Feature hashing for large scale multitask learning},
    author={Kilian Weinberger and Anirban Dasgupta and John Langford and Alex Smola and Josh Attenberg},
    booktitle={Proceedings of the 26th International Conference on Machine Learning (ICML)},
    pages={1113-1120},
    year={2009},
    note={\url{https://arxiv.org/abs/0902.2206}}
}
```
