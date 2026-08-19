# Convolutional layer

Source: [layer_conv2d.go](../../layer_conv2d.go)

2D convolution [[1]](#references) (as in most deep learning frameworks, technically a cross-correlation) over an NCHW input. NCHW describes the dimension order of the tensor: batch size N, channels C, height H, width W. A batch of two RGB images 32x32 would have the shape $[2, 3, 32, 32]$, a single grayscale image 9x8 of the [train_cnn example](../../cmd/examples/train_cnn/main.go) has the shape $[1, 1, 9, 8]$. Output value of the layer:

$$y_{o,i,j} = b_o + \sum_{c=1}^{C} \sum_{m=1}^{K_h} \sum_{n=1}^{K_w} x_{c,\; i \cdot s_h + m,\; j \cdot s_w + n} \cdot k_{o,c,m,n} \tag{1}$$

where $C$ is the number of input channels, $K_h \times K_w$ is the kernel size and $s_h, s_w$ are strides. Padding and dilation are supported as well.

## Shapes

| Tensor | Shape |
| --- | --- |
| Input | $[B, C, H, W]$ |
| `WeightNode` | $[O, C, K_h, K_w]$ |
| Output | $[B, O, H', W']$ |

with $H' = \lfloor (H + 2p_h - d_h(K_h - 1) - 1)/s_h \rfloor + 1$ and $W'$ analogously.

## Backward pass

Let $\delta = \partial L / \partial y$ of shape $[B, O, H', W']$. For unit stride and dilation the gradients are (batch index omitted):

$$\frac{\partial L}{\partial k_{o,c,m,n}} = \sum_{i,j} \delta_{o,i,j} \cdot x_{c,\, i+m,\, j+n} \qquad \frac{\partial L}{\partial b_o} = \sum_{i,j} \delta_{o,i,j} \tag{2}$$

so the kernel gradient is itself a cross-correlation of the input with the output gradient. The input gradient is the transposed operation, a full convolution of $\delta$ with the kernel flipped by 180 degrees:

$$\frac{\partial L}{\partial x_{c,p,q}} = \sum_{o} \sum_{m,n} \delta_{o,\, p-m,\, q-n} \cdot k_{o,c,m,n} \tag{3}$$

Strides and dilation follow the same pattern with the corresponding index arithmetic, Gorgonia derives all of it automatically.

## Usage

```go
layer := &gan.Conv2DLayer{
	WeightNode:   w,
	Activation:   gan.Rectify,
	KernelHeight: 3,
	KernelWidth:  3,
	Padding:      []int{0, 0},
	Stride:       []int{1, 1},
	Dilation:     []int{1, 1},
}
```

## Implementation notes

The computation is delegated to `gorgonia.Conv2d`. Kernel sizes are stored as separate fields, padding/stride/dilation as two-element slices in (height, width) order. See [cmd/examples/train_cnn](../../cmd/examples/train_cnn/main.go) for a complete classifier built with this layer.

## See also

- [Numerical example](../numeric/conv2d.md): the formulas above applied step by step to an RGB 13x9 image with a 3x5 kernel, padding and stride 2.

## References

```bibtex
% [1]
@article{lecun1998gradient,
    title={Gradient-based learning applied to document recognition},
    author={Yann LeCun and L{\'e}on Bottou and Yoshua Bengio and Patrick Haffner},
    journal={Proceedings of the IEEE},
    volume={86},
    number={11},
    pages={2278-2324},
    year={1998},
    note={\url{http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf}}
}
```
