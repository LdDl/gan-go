package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

// LinearLayer Fully connected layer: activate(input × Wᵀ + bias)
type LinearLayer struct {
	WeightNode *gorgonia.Node
	BiasNode   *gorgonia.Node
	Activation ActivationFunc
}

// Fwd Initializates feedforward for provided input.
// Plain matrix multiplication handles 2D input of any batch size,
// batched multiplication is applied for inputs of higher dimensions
func (layer *LinearLayer) Fwd(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	input, err := singleInput("linear", inputs...)
	if err != nil {
		return nil, err
	}
	if layer.WeightNode == nil {
		return nil, fmt.Errorf("Linear layer's weights node is nil")
	}
	tOp, err := gorgonia.Transpose(layer.WeightNode)
	if err != nil {
		return nil, errors.Wrap(err, "Can't transpose weights of layer")
	}
	var layerNonActivated *gorgonia.Node
	if input.Dims() <= 2 {
		layerNonActivated, err = gorgonia.Mul(input, tOp)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiply input and weights of layer")
		}
	} else {
		layerNonActivated, err = gorgonia.BatchedMatMul(input, tOp)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiply input and weights of layer [batched]")
		}
	}
	return addBias(layerNonActivated, layer.BiasNode)
}

// Activate Applies layer's activation function
func (layer *LinearLayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return applyActivation(layer.Activation, input)
}

// Learnables Returns learnable nodes
func (layer *LinearLayer) Learnables() gorgonia.Nodes {
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
func (layer *LinearLayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	if layer.WeightNode == nil {
		return nil, fmt.Errorf("Linear layer has nil weight node")
	}
	return &LinearLayer{
		WeightNode: cloneLearnableTo(g, layer.WeightNode, nameSuffix),
		BiasNode:   cloneLearnableTo(g, layer.BiasNode, nameSuffix),
		Activation: layer.Activation,
	}, nil
}
