package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

// DropoutLayer Applies dropout to the input.
// Has no learnable parameters and no activation.
type DropoutLayer struct {
	Probability float64
}

// Fwd Initializates feedforward for provided input
func (layer *DropoutLayer) Fwd(batchSize int, inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	input, err := singleInput("dropout", inputs...)
	if err != nil {
		return nil, err
	}
	if ok := checkF64ValueInRange(layer.Probability, 0.0, 1.0); !ok {
		return nil, fmt.Errorf("Dropout probability should lie in [0;1] for layer. Got %f", layer.Probability)
	}
	diluted, err := gorgonia.Dropout(input, layer.Probability)
	if err != nil {
		return nil, errors.Wrap(err, "Can't dilute input of layer")
	}
	return diluted, nil
}

// Activate Dropout layer does not imply activation
func (layer *DropoutLayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return input, nil
}

// Learnables Returns learnable nodes. Dropout layer has no learnables.
func (layer *DropoutLayer) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{}
}

// CloneTo Copies layer structure onto the provided graph
func (layer *DropoutLayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	return &DropoutLayer{Probability: layer.Probability}, nil
}
