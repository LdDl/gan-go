package gan_go

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// AvgpoolLayer 2D average pooling layer. Has no learnable parameters.
type AvgpoolLayer struct {
	Activation ActivationFunc

	KernelHeight int
	KernelWidth  int
	Padding      []int
	Stride       []int
}

// Fwd Initializates feedforward for provided input
func (layer *AvgpoolLayer) Fwd(batchSize int, inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	input, err := singleInput("avgpool", inputs...)
	if err != nil {
		return nil, err
	}
	layerNonActivated, err := gorgonia.AveragePool2D(input, tensor.Shape{layer.KernelHeight, layer.KernelWidth}, layer.Padding, layer.Stride)
	if err != nil {
		return nil, errors.Wrap(err, "Can't avgpool[2D] input by kernel of layer")
	}
	return layerNonActivated, nil
}

// Activate Applies layer's activation function
func (layer *AvgpoolLayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return applyActivation(layer.Activation, input)
}

// Learnables Returns learnable nodes. Avgpool layer has no learnables.
func (layer *AvgpoolLayer) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{}
}

// CloneTo Copies layer structure onto the provided graph
func (layer *AvgpoolLayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	return &AvgpoolLayer{
		Activation:   layer.Activation,
		KernelHeight: layer.KernelHeight,
		KernelWidth:  layer.KernelWidth,
		Padding:      layer.Padding,
		Stride:       layer.Stride,
	}, nil
}
