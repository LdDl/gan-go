# Reshape layer

Source: [layer_reshape.go](../../layer_reshape.go)

Reshapes the input tensor to the provided dimensions. The total number of elements must stay the same. The typical use is turning a flat generator output back into an image shape, e.g. $[1, 225]$ into $[1, 1, 15, 15]$ in the [smiley face example](../../cmd/examples/generate_smiley_face/main.go).

The layer has no learnable parameters and no activation, `Activate` returns the input unchanged. For the batch-derived special case see [flatten.md](flatten.md).

## Backward pass

Identical to [flatten.md](flatten.md): the gradient is reshaped back to the input shape, no values are recomputed.

## Usage

```go
layer := &gan.ReshapeLayer{
	Dims: []int{1, 1, 15, 15},
}
```

## Implementation notes

Implemented as a single `gorgonia.Reshape`. Note that reshape results are views sharing memory with the source node, see pitfall 2 in [pitfalls.md](../pitfalls.md) for the constraints this implies inside the library.
