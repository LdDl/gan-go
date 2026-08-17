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

// Fwd Initializates feedforward for provided input
//
// batchSize - batch size. If it's >= 2 then batched multiplication/broadcasting will be applied
func (layer *LinearLayer) Fwd(batchSize int, inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
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
	if batchSize < 2 {
		layerNonActivated, err = gorgonia.Mul(input, tOp)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiply input and weights of layer [batch_size = 1]")
		}
	} else {
		layerNonActivated, err = gorgonia.BatchedMatMul(input, tOp)
		if err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("Can't multiply input and weights of layer [batch_size = %d]", batchSize))
		}
	}
	return addBias(layerNonActivated, layer.BiasNode, batchSize)
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
