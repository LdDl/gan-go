# Flatten layer

Source: [layer_flatten.go](../../layer_flatten.go)

Represents the input tensor as a matrix $[B, N/B]$, where $B$ is the batch size (the first dimension of the input) and $N$ is the total number of elements. The typical use is bridging convolutional feature maps $[B, C, H, W]$ to a [linear](linear.md) layer expecting $[B, C \cdot H \cdot W]$.

The layer has no learnable parameters and no activation, `Activate` returns the input unchanged.

## Backward pass

Reshaping moves no values, so the gradient is the incoming gradient reshaped back to the input shape:

$$\frac{\partial L}{\partial x} = \operatorname{reshape}(\delta, \operatorname{shape}(x)) \tag{1}$$

## Usage

```go
layer := &gan.FlattenLayer{}
```

## Implementation notes

Implemented as a single `gorgonia.Reshape`. For arbitrary target dimensions see [reshape.md](reshape.md).
