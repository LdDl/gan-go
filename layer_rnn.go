package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// RNNLayer Vanilla recurrent layer.
//
// For hidden size H:
//
//	InputWeightNode holds W of shape [input_features, H]
//	HiddenWeightNode holds U of shape [H, H]
//	BiasNode (optional) holds b of shape [1, H]
//
// For time step t hidden state is computed as:
//
//	h_t = activation(x_t * W + h_prev * U + b)
//
// where activation should be Tanh most of times (used if nil).
//
// Input must be a tensor of shape [sequence, input_features] or [sequence, batch, input_features].
// Output is a tensor of hidden states for every time step: [sequence, H] or [sequence, batch, H] respectively.
//
// Initial hidden state is a zero-valued node created automatically.
// Custom one could be provided either via InitialHiddenNode field or as second input node of Fwd() call.
// Expected shape is [batch, H].
type RNNLayer struct {
	InputWeightNode  *gorgonia.Node
	HiddenWeightNode *gorgonia.Node
	BiasNode         *gorgonia.Node

	InitialHiddenNode *gorgonia.Node

	HiddenSize int
	/* Hidden state activation. Should be Tanh most of times (used if nil) */
	Activation ActivationFunc

	finalHiddenNode *gorgonia.Node
}

// FinalHidden Returns reference to hidden state node of the last time step. It is set by Fwd() call
func (layer *RNNLayer) FinalHidden() *gorgonia.Node {
	return layer.finalHiddenNode
}

// Fwd Initializates feedforward for provided input
//
// inputs - either single input node or (input, initial hidden state) pair
func (layer *RNNLayer) Fwd(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	var input, hiddenState *gorgonia.Node
	switch len(inputs) {
	case 1:
		input = inputs[0]
		hiddenState = layer.InitialHiddenNode
	case 2:
		input = inputs[0]
		hiddenState = inputs[1]
	default:
		return nil, fmt.Errorf("Layer of type 'rnn' can handle either 1 or 2 input nodes, got %d", len(inputs))
	}
	if layer.InputWeightNode == nil {
		return nil, fmt.Errorf("RNN layer's input weights node is nil")
	}
	if layer.HiddenWeightNode == nil {
		return nil, fmt.Errorf("RNN layer's hidden weights node is nil")
	}
	if layer.HiddenSize < 1 {
		return nil, fmt.Errorf("RNN layer's hidden size should be positive, got %d", layer.HiddenSize)
	}
	hiddenSize := layer.HiddenSize
	if got := layer.InputWeightNode.Shape()[1]; got != hiddenSize {
		return nil, fmt.Errorf("RNN layer's input weights node should have shape [input_features, %d], got %v", hiddenSize, layer.InputWeightNode.Shape())
	}
	if got := layer.HiddenWeightNode.Shape(); got[0] != hiddenSize || got[1] != hiddenSize {
		return nil, fmt.Errorf("RNN layer's hidden weights node should have shape [%d, %d], got %v", hiddenSize, hiddenSize, got)
	}
	activation := layer.Activation
	if activation == nil {
		activation = Tanh
	}

	// Single code path for both [sequence, features] and [sequence, batch, features] inputs
	squeezeOutput := false
	if input.Dims() == 2 {
		squeezeOutput = true
		reshaped, err := gorgonia.Reshape(input, tensor.Shape{input.Shape()[0], 1, input.Shape()[1]})
		if err != nil {
			return nil, errors.Wrap(err, "Can't add batch dimension to RNN layer's input")
		}
		input = reshaped
	}
	if input.Dims() != 3 {
		return nil, fmt.Errorf("RNN layer's input should have shape [sequence, input_features] or [sequence, batch, input_features], got %v", input.Shape())
	}
	sequenceLength := input.Shape()[0]
	batch := input.Shape()[1]

	// Zero-valued initial state unless custom one is provided.
	// Name must be unique in scope of graph: Gorgonia hashes input nodes by type, shape and name only
	if hiddenState == nil {
		hiddenState = gorgonia.NewMatrix(input.Graph(), input.Dtype(), gorgonia.WithShape(batch, hiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName(fmt.Sprintf("rnn_%d_initial_hidden", input.ID())))
	}

	outputs := make([]*gorgonia.Node, sequenceLength)
	for t := 0; t < sequenceLength; t++ {
		// x_t of shape [batch, input_features]
		xt, err := gorgonia.Slice(input, gorgonia.S(t), nil, nil)
		if err != nil {
			return nil, errors.Wrapf(err, "Can't slice time step %d of RNN layer's input of shape %v", t, input.Shape())
		}
		inputProjection, err := gorgonia.Mul(xt, layer.InputWeightNode)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiply time step input and input weights of RNN layer")
		}
		hiddenProjection, err := gorgonia.Mul(hiddenState, layer.HiddenWeightNode)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiply previous hidden state and hidden weights of RNN layer")
		}
		preActivation, err := gorgonia.Add(inputProjection, hiddenProjection)
		if err != nil {
			return nil, errors.Wrap(err, "Can't sum input and hidden projections of RNN layer")
		}
		preActivation, err = addBias(preActivation, layer.BiasNode)
		if err != nil {
			return nil, errors.Wrap(err, "Can't add bias to pre-activation of RNN layer")
		}
		hiddenState, err = activation(preActivation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't apply activation function to hidden state of RNN layer")
		}
		outputs[t], err = gorgonia.Reshape(hiddenState, tensor.Shape{1, batch, hiddenSize})
		if err != nil {
			return nil, errors.Wrap(err, "Can't add time dimension to hidden state of RNN layer")
		}
	}
	layer.finalHiddenNode = hiddenState

	layerNonActivated, err := gorgonia.Concat(0, outputs...)
	if err != nil {
		return nil, errors.Wrap(err, "Can't concatenate hidden states of RNN layer")
	}
	if squeezeOutput {
		layerNonActivated, err = gorgonia.Reshape(layerNonActivated, tensor.Shape{sequenceLength, hiddenSize})
		if err != nil {
			return nil, errors.Wrap(err, "Can't squeeze batch dimension of RNN layer's output")
		}
	}
	return layerNonActivated, nil
}

// Activate RNN layer does not imply activation of output: activation function is applied inside of the recurrence
func (layer *RNNLayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return input, nil
}

// Learnables Returns learnable nodes
func (layer *RNNLayer) Learnables() gorgonia.Nodes {
	learnables := make(gorgonia.Nodes, 0, 3)
	if layer.InputWeightNode != nil {
		learnables = append(learnables, layer.InputWeightNode)
	}
	if layer.HiddenWeightNode != nil {
		learnables = append(learnables, layer.HiddenWeightNode)
	}
	if layer.BiasNode != nil {
		learnables = append(learnables, layer.BiasNode)
	}
	return learnables
}

// CloneTo Copies layer structure onto the provided graph. Learnables of the copy share underlying tensors with the source layer
func (layer *RNNLayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	if layer.InputWeightNode == nil {
		return nil, fmt.Errorf("RNN layer has nil input weights node")
	}
	if layer.HiddenWeightNode == nil {
		return nil, fmt.Errorf("RNN layer has nil hidden weights node")
	}
	return &RNNLayer{
		InputWeightNode:   cloneLearnableTo(g, layer.InputWeightNode, nameSuffix),
		HiddenWeightNode:  cloneLearnableTo(g, layer.HiddenWeightNode, nameSuffix),
		BiasNode:          cloneLearnableTo(g, layer.BiasNode, nameSuffix),
		InitialHiddenNode: cloneLearnableTo(g, layer.InitialHiddenNode, nameSuffix),
		HiddenSize:        layer.HiddenSize,
		Activation:        layer.Activation,
	}, nil
}
