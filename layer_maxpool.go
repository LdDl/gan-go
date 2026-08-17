package gan_go

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// MaxpoolLayer 2D max-pooling layer. Has no learnable parameters.
type MaxpoolLayer struct {
	Activation ActivationFunc

	KernelHeight int
	KernelWidth  int
	Padding      []int
	Stride       []int
}

// Fwd Initializates feedforward for provided input
func (layer *MaxpoolLayer) Fwd(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	input, err := singleInput("maxpool", inputs...)
	if err != nil {
		return nil, err
	}
	layerNonActivated, err := gorgonia.MaxPool2D(input, tensor.Shape{layer.KernelHeight, layer.KernelWidth}, layer.Padding, layer.Stride)
	if err != nil {
		return nil, errors.Wrap(err, "Can't maxpool[2D] input by kernel of layer")
	}
	return layerNonActivated, nil
}

// Activate Applies layer's activation function
func (layer *MaxpoolLayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return applyActivation(layer.Activation, input)
}

// Learnables Returns learnable nodes. Maxpool layer has no learnables.
func (layer *MaxpoolLayer) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{}
}

// CloneTo Copies layer structure onto the provided graph
func (layer *MaxpoolLayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	return &MaxpoolLayer{
		Activation:   layer.Activation,
		KernelHeight: layer.KernelHeight,
		KernelWidth:  layer.KernelWidth,
		Padding:      layer.Padding,
		Stride:       layer.Stride,
	}, nil
}
