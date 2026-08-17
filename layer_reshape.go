package gan_go

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

// ReshapeLayer Reshapes input tensor to provided dimensions.
// Has no learnable parameters and no activation.
type ReshapeLayer struct {
	Dims []int
}

// Fwd Initializates feedforward for provided input
func (layer *ReshapeLayer) Fwd(batchSize int, inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	input, err := singleInput("reshape", inputs...)
	if err != nil {
		return nil, err
	}
	reshaped, err := gorgonia.Reshape(input, layer.Dims)
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape input of layer")
	}
	return reshaped, nil
}

// Activate Reshape layer does not imply activation
func (layer *ReshapeLayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return input, nil
}

// Learnables Returns learnable nodes. Reshape layer has no learnables.
func (layer *ReshapeLayer) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{}
}

// CloneTo Copies layer structure onto the provided graph
func (layer *ReshapeLayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	return &ReshapeLayer{Dims: layer.Dims}, nil
}
