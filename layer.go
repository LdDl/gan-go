package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Layer Interface for a single layer of a neural network.
//
// Fwd - Builds forward pass for provided inputs and returns non-activated output node.
//
//	Batch size is derived from shapes of the inputs, first dimension is considered to be the batch one.
//
// Activate - Applies layer's activation function to non-activated output node.
//
//	Layers which do not imply activation (e.g. Flatten/Reshape/Dropout/Embedding) just return input node as is.
//
// Learnables - Returns nodes containing learnable parameters (could be empty)
// CloneTo - Copies layer structure onto the provided graph.
//
//	Learnable nodes of the copy are new nodes bound to THE SAME underlying tensors,
//	see notes for NewGAN in gan.go about this shared-memory trick.
type Layer interface {
	Fwd(inputs ...*gorgonia.Node) (*gorgonia.Node, error)
	Activate(input *gorgonia.Node) (*gorgonia.Node, error)
	Learnables() gorgonia.Nodes
	CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error)
}

// singleInput Helper for layers which can handle only one input node
func singleInput(layerType string, inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("There are no input nodes for layer of type '%s'", layerType)
	}
	if len(inputs) > 1 {
		return nil, fmt.Errorf("Layer of type '%s' can handle only 1 input node, got %d", layerType, len(inputs))
	}
	return inputs[0], nil
}

// addBias Helper adding bias node to non-activated output of a layer.
// If bias node is nil then non-activated output is returned as is.
// If batch size (first dimension of the non-activated output) is >= 2 then broadcast function will be applied
func addBias(layerNonActivated, bias *gorgonia.Node) (*gorgonia.Node, error) {
	if bias == nil {
		return layerNonActivated, nil
	}
	batchSize := layerNonActivated.Shape()[0]
	if batchSize < 2 {
		withBias, err := gorgonia.Add(layerNonActivated, bias)
		if err != nil {
			return nil, errors.Wrap(err, "Can't add bias to non-activated output of a layer")
		}
		return withBias, nil
	}
	withBias, err := gorgonia.BroadcastAdd(layerNonActivated, bias, nil, []byte{0})
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("Can't add [in broadcast term with batch_size = %d] bias to non-activated output of a layer", batchSize))
	}
	return withBias, nil
}

// applyActivation Helper applying activation function. Nil function is considered to be NoActivation.
func applyActivation(f ActivationFunc, input *gorgonia.Node) (*gorgonia.Node, error) {
	if f == nil {
		return input, nil
	}
	return f(input)
}

// cloneLearnableTo Creates node on the provided graph copying dtype/shape/name (with suffix) of the source node.
//
// Important: gorgonia.WithValue(...) binds the very same tensor as the source node has.
// Backing memory is shared, so in-place updates done by a solver on the source graph
// automatically propagate to the copy. See notes for NewGAN in gan.go.
func cloneLearnableTo(g *gorgonia.ExprGraph, node *gorgonia.Node, nameSuffix string) *gorgonia.Node {
	if node == nil {
		return nil
	}
	return gorgonia.NewTensor(g, node.Dtype(), node.Dims(), gorgonia.WithShape(node.Shape()...), gorgonia.WithName(node.Name()+nameSuffix), gorgonia.WithValue(node.Value()))
}

// sliceGate Extracts single gate (columns [idx*hiddenSize; (idx+1)*hiddenSize)) of a recurrent layer
// and applies activation function to it.
//
// Three implementation notes:
// 1. Slice of a node is a view sharing memory with the sliced node. Binary operations of Gorgonia
// may write their result into the buffer of an operand, corrupting the source tensor,
// so views are passed to unary operations only.
// 2. Sliced gate is reshaped explicitly since slicing range of width 1 collapses the dimension.
// 3. Order matters: reshape must be applied BEFORE the activation function.
// The reversed order (slice, activation, reshape) produces incorrect gradients in the backward
// pass of Gorgonia, which is verified by numerical gradient checks in the tests
func sliceGate(gates *gorgonia.Node, idx, hiddenSize int, activation ActivationFunc) (*gorgonia.Node, error) {
	batch := gates.Shape()[0]
	gate, err := gorgonia.Slice(gates, nil, gorgonia.S(idx*hiddenSize, (idx+1)*hiddenSize))
	if err != nil {
		return nil, errors.Wrap(err, "Can't slice gate")
	}
	reshaped, err := gorgonia.Reshape(gate, tensor.Shape{batch, hiddenSize})
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape sliced gate")
	}
	return activation(reshaped)
}

func checkF64ValueInRange(input, min, max float64) bool {
	if input < min || input > max {
		return false
	}
	return true
}
