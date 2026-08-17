package gan_go

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// FlattenLayer Represents input tensor as [batchSize x Total number of elements in tensor / batchSize].
// Has no learnable parameters and no activation.
type FlattenLayer struct {
}

// Fwd Initializates feedforward for provided input
func (layer *FlattenLayer) Fwd(batchSize int, inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	input, err := singleInput("flatten", inputs...)
	if err != nil {
		return nil, err
	}
	flatten, err := gorgonia.Reshape(input, tensor.Shape{batchSize, input.Shape().TotalSize() / batchSize})
	if err != nil {
		return nil, errors.Wrap(err, "Can't flatten input of layer")
	}
	return flatten, nil
}

// Activate Flatten layer does not imply activation
func (layer *FlattenLayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return input, nil
}

// Learnables Returns learnable nodes. Flatten layer has no learnables.
func (layer *FlattenLayer) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{}
}

// CloneTo Copies layer structure onto the provided graph
func (layer *FlattenLayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	return &FlattenLayer{}, nil
}
