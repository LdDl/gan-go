# Maxpool layer

Source: [layer_maxpool.go](../../layer_maxpool.go)

2D max pooling [[1]](#references) over an NCHW input (dimension order: batch, channels, height, width, see [conv2d.md](conv2d.md)): every output element is the maximum of a $K_h \times K_w$ window of the input channel:

$$y_{c,i,j} = \max_{0 \le m < K_h,\; 0 \le n < K_w} x_{c,\; i \cdot s_h + m,\; j \cdot s_w + n} \tag{1}$$

The layer has no learnable parameters. See [avgpool.md](avgpool.md) for the averaging counterpart.

## Backward pass

The gradient is routed to the argmax: every window passes its incoming gradient $\delta_{c,i,j}$ to the single input element which produced the maximum, all other elements of the window receive zero:

$$\frac{\partial L}{\partial x_{c,u,v}} = \sum_{(i,j):\; (u,v) = \operatorname{argmax}\; \text{of window} (i,j)} \delta_{c,i,j} \tag{2}$$

With overlapping windows (stride smaller than kernel) one input element can win several windows and accumulates their gradients.

## Usage

```go
layer := &gan.MaxpoolLayer{
	KernelHeight: 2,
	KernelWidth:  2,
	Padding:      []int{0, 0},
	Stride:       []int{2, 2},
}
```

## Implementation notes

The computation is delegated to `gorgonia.MaxPool2D`. An optional `Activation` field is applied to the pooled output (identity when nil).

## References

```bibtex
% [1]
@inproceedings{boureau2010theoretical,
    title={A theoretical analysis of feature pooling in visual recognition},
    author={Y-Lan Boureau and Jean Ponce and Yann LeCun},
    booktitle={Proceedings of the 27th International Conference on Machine Learning (ICML)},
    pages={111-118},
    year={2010},
    note={\url{https://icml.cc/Conferences/2010/papers/638.pdf}}
}
```
