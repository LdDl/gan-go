package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// EmbeddingLayer Looks up embeddings (rows of weight node) for integer input indices.
// Input must be a vector of type Int. Has no activation.
type EmbeddingLayer struct {
	WeightNode    *gorgonia.Node
	EmbeddingSize int
}

// Fwd Initializates feedforward for provided input
func (layer *EmbeddingLayer) Fwd(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	input, err := singleInput("embedding", inputs...)
	if err != nil {
		return nil, err
	}
	if layer.WeightNode == nil {
		return nil, fmt.Errorf("Embedding layer's weights node is nil")
	}
	if input.Type().String() != "Vector int" {
		return nil, fmt.Errorf("Layer is implemented for type 'Int' not for '%s'", input.Type().String())
	}
	inputLength := input.Shape().TotalSize()
	tmpFlatten, err := gorgonia.Reshape(input, tensor.Shape{inputLength})
	if err != nil {
		return nil, errors.Wrap(err, "Can't flatten input of layer [temporary]")
	}
	tmpEmbedding, err := gorgonia.ByIndices(layer.WeightNode, tmpFlatten, 0)
	if err != nil {
		return nil, errors.Wrap(err, "Can't embedd input of layer [temporary]")
	}
	embedding, err := gorgonia.Reshape(tmpEmbedding, append(input.Shape(), layer.EmbeddingSize))
	if err != nil {
		return nil, errors.Wrap(err, "Can't embedd input of layer")
	}
	return embedding, nil
}

// Activate Embedding layer does not imply activation
func (layer *EmbeddingLayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return input, nil
}

// Learnables Returns learnable nodes
func (layer *EmbeddingLayer) Learnables() gorgonia.Nodes {
	if layer.WeightNode != nil {
		return gorgonia.Nodes{layer.WeightNode}
	}
	return gorgonia.Nodes{}
}

// CloneTo Copies layer structure onto the provided graph. Learnables of the copy share underlying tensors with the source layer.
func (layer *EmbeddingLayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	if layer.WeightNode == nil {
		return nil, fmt.Errorf("Embedding layer has nil weight node")
	}
	return &EmbeddingLayer{
		WeightNode:    cloneLearnableTo(g, layer.WeightNode, nameSuffix),
		EmbeddingSize: layer.EmbeddingSize,
	}, nil
}
