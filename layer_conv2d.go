package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Conv2DLayer 2D convolutional layer: activate(conv2d(input, W) + bias)
type Conv2DLayer struct {
	WeightNode *gorgonia.Node
	BiasNode   *gorgonia.Node
	Activation ActivationFunc

	KernelHeight int
	KernelWidth  int
	Padding      []int
	Stride       []int
	Dilation     []int
}

// Fwd Initializates feedforward for provided input
func (layer *Conv2DLayer) Fwd(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	input, err := singleInput("conv2d", inputs...)
	if err != nil {
		return nil, err
	}
	if layer.WeightNode == nil {
		return nil, fmt.Errorf("Convolutional layer's weights node is nil")
	}
	layerNonActivated, err := gorgonia.Conv2d(input, layer.WeightNode, tensor.Shape{layer.KernelHeight, layer.KernelWidth}, layer.Padding, layer.Stride, layer.Dilation)
	if err != nil {
		return nil, errors.Wrap(err, "Can't convolve[2D] input by kernel of layer")
	}
	return addBias(layerNonActivated, layer.BiasNode)
}

// Activate Applies layer's activation function
func (layer *Conv2DLayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return applyActivation(layer.Activation, input)
}

// Learnables Returns learnable nodes
func (layer *Conv2DLayer) Learnables() gorgonia.Nodes {
	learnables := make(gorgonia.Nodes, 0, 2)
	if layer.WeightNode != nil {
		learnables = append(learnables, layer.WeightNode)
	}
	if layer.BiasNode != nil {
		learnables = append(learnables, layer.BiasNode)
	}
	return learnables
}

// CloneTo Copies layer structure onto the provided graph. Learnables of the copy share underlying tensors with the source layer.
func (layer *Conv2DLayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	if layer.WeightNode == nil {
		return nil, fmt.Errorf("Convolutional layer has nil weight node")
	}
	return &Conv2DLayer{
		WeightNode:   cloneLearnableTo(g, layer.WeightNode, nameSuffix),
		BiasNode:     cloneLearnableTo(g, layer.BiasNode, nameSuffix),
		Activation:   layer.Activation,
		KernelHeight: layer.KernelHeight,
		KernelWidth:  layer.KernelWidth,
		Padding:      layer.Padding,
		Stride:       layer.Stride,
		Dilation:     layer.Dilation,
	}, nil
}
