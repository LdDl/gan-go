# AvgPool layer

Source: [layer_avgpool.go](../../layer_avgpool.go)

2D average pooling [[1, 2]](#references) over an NCHW input (dimension order: batch, channels, height, width, see [conv2d.md](conv2d.md)): every output element is the mean of a $K_h \times K_w$ window of the input channel:

$$y_{c,i,j} = \frac{1}{K_h K_w} \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} x_{c,\; i \cdot s_h + m,\; j \cdot s_w + n} \tag{1}$$

The layer has no learnable parameters and mirrors [maxpool.md](maxpool.md) in every option.

## Backward pass

Every input element of a window receives an equal share of the incoming gradient:

$$\frac{\partial L}{\partial x_{c,u,v}} = \sum_{(i,j):\; (u,v) \in \text{window}(i,j)} \frac{\delta_{c,i,j}}{K_h K_w} \tag{2}$$

Compare with [maxpool.md](maxpool.md), where the whole gradient goes to the argmax only: average pooling spreads the learning signal, max pooling concentrates it.

## Usage

```go
layer := &gan.AvgpoolLayer{
	KernelHeight: 2,
	KernelWidth:  2,
	Padding:      []int{0, 0},
	Stride:       []int{2, 2},
}
```

## Implementation notes

The computation is delegated to `gorgonia.AveragePool2D` (available since Gorgonia v0.9.18). Output is verified against hand computed window averages in [layer_avgpool_test.go](../../layer_avgpool_test.go).

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
% [2]
@inproceedings{boureau2010theoretical,
    title={A theoretical analysis of feature pooling in visual recognition},
    author={Y-Lan Boureau and Jean Ponce and Yann LeCun},
    booktitle={Proceedings of the 27th International Conference on Machine Learning (ICML)},
    pages={111-118},
    year={2010},
    note={\url{https://icml.cc/Conferences/2010/papers/638.pdf}}
}
```
