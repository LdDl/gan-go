package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// GRULayer Gated recurrent unit layer (formulation of Cho et al., 2014).
//
// All three gates (reset, update, candidate) are packed into single weight matrices,
// so for hidden size H:
//
//	InputWeightNode holds W of shape [input_features, 3*H]
//	HiddenWeightNode holds U of shape [H, 3*H]
//	BiasNode (optional) holds b of shape [1, 3*H], applied to the input projection
//
// For time step t (gate columns order is reset, update, candidate):
//
//	r_t = sigmoid((x_t * W + b)_r + (h_prev * U)_r)
//	z_t = sigmoid((x_t * W + b)_z + (h_prev * U)_z)
//	n_t = tanh((x_t * W + b)_n + ((r_t ⊙ h_prev) * U)_n)
//	h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_prev
//
// where sigmoid could be replaced via RecurrentActivation and tanh via Activation.
//
// Input must be a tensor of shape [sequence, input_features] or [sequence, batch, input_features].
// Output is a tensor of hidden states for every time step: [sequence, H] or [sequence, batch, H] respectively.
//
// Initial hidden state is a zero-valued node created automatically.
// Custom one could be provided either via InitialHiddenNode field or as second input node of Fwd() call.
// Expected shape is [batch, H].
type GRULayer struct {
	InputWeightNode  *gorgonia.Node
	HiddenWeightNode *gorgonia.Node
	BiasNode         *gorgonia.Node

	InitialHiddenNode *gorgonia.Node

	HiddenSize int
	/* Candidate activation. Should be Tanh most of times (used if nil) */
	Activation ActivationFunc
	/* Reset and update gates activation. Should be Sigmoid most of times (used if nil) */
	RecurrentActivation ActivationFunc

	finalHiddenNode *gorgonia.Node
}

// FinalHidden Returns reference to hidden state node of the last time step. It is set by Fwd() call
func (layer *GRULayer) FinalHidden() *gorgonia.Node {
	return layer.finalHiddenNode
}

// Fwd Initializates feedforward for provided input
//
// inputs - either single input node or (input, initial hidden state) pair
func (layer *GRULayer) Fwd(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	var input, hiddenState *gorgonia.Node
	switch len(inputs) {
	case 1:
		input = inputs[0]
		hiddenState = layer.InitialHiddenNode
	case 2:
		input = inputs[0]
		hiddenState = inputs[1]
	default:
		return nil, fmt.Errorf("Layer of type 'gru' can handle either 1 or 2 input nodes, got %d", len(inputs))
	}
	if layer.InputWeightNode == nil {
		return nil, fmt.Errorf("GRU layer's input weights node is nil")
	}
	if layer.HiddenWeightNode == nil {
		return nil, fmt.Errorf("GRU layer's hidden weights node is nil")
	}
	if layer.HiddenSize < 1 {
		return nil, fmt.Errorf("GRU layer's hidden size should be positive, got %d", layer.HiddenSize)
	}
	hiddenSize := layer.HiddenSize
	if got := layer.InputWeightNode.Shape()[1]; got != 3*hiddenSize {
		return nil, fmt.Errorf("GRU layer's input weights node should have shape [input_features, %d], got %v", 3*hiddenSize, layer.InputWeightNode.Shape())
	}
	if got := layer.HiddenWeightNode.Shape(); got[0] != hiddenSize || got[1] != 3*hiddenSize {
		return nil, fmt.Errorf("GRU layer's hidden weights node should have shape [%d, %d], got %v", hiddenSize, 3*hiddenSize, got)
	}
	candidateActivation := layer.Activation
	if candidateActivation == nil {
		candidateActivation = Tanh
	}
	gatesActivation := layer.RecurrentActivation
	if gatesActivation == nil {
		gatesActivation = Sigmoid
	}

	// Single code path for both [sequence, features] and [sequence, batch, features] inputs
	squeezeOutput := false
	if input.Dims() == 2 {
		squeezeOutput = true
		reshaped, err := gorgonia.Reshape(input, tensor.Shape{input.Shape()[0], 1, input.Shape()[1]})
		if err != nil {
			return nil, errors.Wrap(err, "Can't add batch dimension to GRU layer's input")
		}
		input = reshaped
	}
	if input.Dims() != 3 {
		return nil, fmt.Errorf("GRU layer's input should have shape [sequence, input_features] or [sequence, batch, input_features], got %v", input.Shape())
	}
	sequenceLength := input.Shape()[0]
	batch := input.Shape()[1]

	// Zero-valued initial state unless custom one is provided.
	// Name must be unique in scope of graph: Gorgonia hashes input nodes by type, shape and name only
	if hiddenState == nil {
		hiddenState = gorgonia.NewMatrix(input.Graph(), input.Dtype(), gorgonia.WithShape(batch, hiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName(fmt.Sprintf("gru_%d_initial_hidden", input.ID())))
	}
	// Zero-valued node for detaching stored outputs from hidden state buffers.
	// Tape machine of Gorgonia is free to reuse a buffer of a node for the result of the last
	// operation reading that node (in-place optimization). Element-wise operations of the next
	// time step do read the hidden state, while the stored output references its buffer through
	// a reshape view invisible to the liveness analysis. Summation with zeros materializes
	// a standalone copy, so stored outputs survive such buffer reuse
	outputAnchor := gorgonia.NewMatrix(input.Graph(), input.Dtype(), gorgonia.WithShape(batch, hiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName(fmt.Sprintf("gru_%d_output_anchor", input.ID())))

	outputs := make([]*gorgonia.Node, sequenceLength)
	for t := 0; t < sequenceLength; t++ {
		// x_t of shape [batch, input_features]
		xt, err := gorgonia.Slice(input, gorgonia.S(t), nil, nil)
		if err != nil {
			return nil, errors.Wrapf(err, "Can't slice time step %d of GRU layer's input of shape %v", t, input.Shape())
		}
		inputProjection, err := gorgonia.Mul(xt, layer.InputWeightNode)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiply time step input and input weights of GRU layer")
		}
		inputProjection, err = addBias(inputProjection, layer.BiasNode)
		if err != nil {
			return nil, errors.Wrap(err, "Can't add bias to input projection of GRU layer")
		}
		hiddenProjection, err := gorgonia.Mul(hiddenState, layer.HiddenWeightNode)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiply previous hidden state and hidden weights of GRU layer")
		}
		// Reset and update gates.
		// Note: hidden projection is intentionally the first operand. Some element-wise operations
		// of Gorgonia write their result into the buffer of the first operand, so a node which is
		// read again later (input projection here) must not be placed first
		gates, err := gorgonia.Add(hiddenProjection, inputProjection)
		if err != nil {
			return nil, errors.Wrap(err, "Can't sum input and hidden projections of GRU layer")
		}
		resetGate, err := sliceGate(gates, 0, hiddenSize, gatesActivation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't extract reset gate of GRU layer")
		}
		updateGate, err := sliceGate(gates, 1, hiddenSize, gatesActivation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't extract update gate of GRU layer")
		}
		// Candidate: reset gate is applied to the previous hidden state before the projection
		hiddenGated, err := gorgonia.HadamardProd(resetGate, hiddenState)
		if err != nil {
			return nil, errors.Wrap(err, "Can't apply reset gate to previous hidden state of GRU layer")
		}
		hiddenGatedProjection, err := gorgonia.Mul(hiddenGated, layer.HiddenWeightNode)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiply gated hidden state and hidden weights of GRU layer")
		}
		candidateSum, err := gorgonia.Add(hiddenGatedProjection, inputProjection)
		if err != nil {
			return nil, errors.Wrap(err, "Can't sum input and gated hidden projections of GRU layer")
		}
		candidate, err := sliceGate(candidateSum, 2, hiddenSize, candidateActivation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't extract candidate of GRU layer")
		}
		// h_t = (1 - z) ⊙ n + z ⊙ h_prev, computed as n + z ⊙ (h_prev - n)
		hiddenSubCandidate, err := gorgonia.Sub(hiddenState, candidate)
		if err != nil {
			return nil, errors.Wrap(err, "Can't subtract candidate from previous hidden state of GRU layer")
		}
		keptPart, err := gorgonia.HadamardProd(updateGate, hiddenSubCandidate)
		if err != nil {
			return nil, errors.Wrap(err, "Can't apply update gate of GRU layer")
		}
		hiddenState, err = gorgonia.Add(candidate, keptPart)
		if err != nil {
			return nil, errors.Wrap(err, "Can't update hidden state of GRU layer")
		}
		outputCopy, err := gorgonia.Add(hiddenState, outputAnchor)
		if err != nil {
			return nil, errors.Wrap(err, "Can't copy hidden state of GRU layer for storing")
		}
		outputs[t], err = gorgonia.Reshape(outputCopy, tensor.Shape{1, batch, hiddenSize})
		if err != nil {
			return nil, errors.Wrap(err, "Can't add time dimension to hidden state of GRU layer")
		}
	}
	layer.finalHiddenNode = hiddenState

	layerNonActivated, err := gorgonia.Concat(0, outputs...)
	if err != nil {
		return nil, errors.Wrap(err, "Can't concatenate hidden states of GRU layer")
	}
	if squeezeOutput {
		layerNonActivated, err = gorgonia.Reshape(layerNonActivated, tensor.Shape{sequenceLength, hiddenSize})
		if err != nil {
			return nil, errors.Wrap(err, "Can't squeeze batch dimension of GRU layer's output")
		}
	}
	return layerNonActivated, nil
}

// Activate GRU layer does not imply activation of output: activation functions are applied inside of the cell
func (layer *GRULayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return input, nil
}

// Learnables Returns learnable nodes
func (layer *GRULayer) Learnables() gorgonia.Nodes {
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
func (layer *GRULayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	if layer.InputWeightNode == nil {
		return nil, fmt.Errorf("GRU layer has nil input weights node")
	}
	if layer.HiddenWeightNode == nil {
		return nil, fmt.Errorf("GRU layer has nil hidden weights node")
	}
	return &GRULayer{
		InputWeightNode:     cloneLearnableTo(g, layer.InputWeightNode, nameSuffix),
		HiddenWeightNode:    cloneLearnableTo(g, layer.HiddenWeightNode, nameSuffix),
		BiasNode:            cloneLearnableTo(g, layer.BiasNode, nameSuffix),
		InitialHiddenNode:   cloneLearnableTo(g, layer.InitialHiddenNode, nameSuffix),
		HiddenSize:          layer.HiddenSize,
		Activation:          layer.Activation,
		RecurrentActivation: layer.RecurrentActivation,
	}, nil
}
