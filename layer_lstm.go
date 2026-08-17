package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// LSTMLayer Long short-term memory layer.
//
// All four gates (input, forget, cell candidate, output) are packed into single weight matrices,
// so for hidden size H:
//
//	InputWeightNode holds W of shape [input_features, 4*H]
//	HiddenWeightNode holds U of shape [H, 4*H]
//	BiasNode (optional) holds b of shape [1, 4*H]
//
// For time step t gates are computed as slices of (x_t * W + h_prev * U + b), then:
//
//	c_t = sigmoid(f) ⊙ c_prev + sigmoid(i) ⊙ tanh(g)
//	h_t = sigmoid(o) ⊙ tanh(c_t)
//
// where sigmoid could be replaced via RecurrentActivation and tanh via Activation.
//
// Input must be a tensor of shape [sequence, input_features] or [sequence, batch, input_features].
// Output is a tensor of hidden states for every time step: [sequence, H] or [sequence, batch, H] respectively.
//
// Initial hidden state and initial cell state are zero-valued nodes created automatically.
// Custom ones could be provided either via InitialHiddenNode/InitialCellNode fields
// or as second and third input nodes of Fwd() call. Expected shape is [batch, H].
type LSTMLayer struct {
	InputWeightNode  *gorgonia.Node
	HiddenWeightNode *gorgonia.Node
	BiasNode         *gorgonia.Node

	InitialHiddenNode *gorgonia.Node
	InitialCellNode   *gorgonia.Node

	HiddenSize int
	/* Cell candidate and cell output activation. Should be Tanh most of times (used if nil) */
	Activation ActivationFunc
	/* Gates activation. Should be Sigmoid most of times (used if nil) */
	RecurrentActivation ActivationFunc

	finalHiddenNode *gorgonia.Node
	finalCellNode   *gorgonia.Node
}

// FinalHidden Returns reference to hidden state node of the last time step. It is set by Fwd() call
func (layer *LSTMLayer) FinalHidden() *gorgonia.Node {
	return layer.finalHiddenNode
}

// FinalCell Returns reference to cell state node of the last time step. It is set by Fwd() call
func (layer *LSTMLayer) FinalCell() *gorgonia.Node {
	return layer.finalCellNode
}

// Fwd Initializates feedforward for provided input
//
// inputs - either single input node or (input, initial hidden state, initial cell state) triple
// batchSize - not used by this layer type since batch size is derived from the input shape
func (layer *LSTMLayer) Fwd(batchSize int, inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	var input, hiddenState, cellState *gorgonia.Node
	switch len(inputs) {
	case 1:
		input = inputs[0]
		hiddenState = layer.InitialHiddenNode
		cellState = layer.InitialCellNode
	case 3:
		input = inputs[0]
		hiddenState = inputs[1]
		cellState = inputs[2]
	default:
		return nil, fmt.Errorf("Layer of type 'lstm' can handle either 1 or 3 input nodes, got %d", len(inputs))
	}
	if layer.InputWeightNode == nil {
		return nil, fmt.Errorf("LSTM layer's input weights node is nil")
	}
	if layer.HiddenWeightNode == nil {
		return nil, fmt.Errorf("LSTM layer's hidden weights node is nil")
	}
	if layer.HiddenSize < 1 {
		return nil, fmt.Errorf("LSTM layer's hidden size should be positive, got %d", layer.HiddenSize)
	}
	hiddenSize := layer.HiddenSize
	if got := layer.InputWeightNode.Shape()[1]; got != 4*hiddenSize {
		return nil, fmt.Errorf("LSTM layer's input weights node should have shape [input_features, %d], got %v", 4*hiddenSize, layer.InputWeightNode.Shape())
	}
	if got := layer.HiddenWeightNode.Shape(); got[0] != hiddenSize || got[1] != 4*hiddenSize {
		return nil, fmt.Errorf("LSTM layer's hidden weights node should have shape [%d, %d], got %v", hiddenSize, 4*hiddenSize, got)
	}
	cellActivation := layer.Activation
	if cellActivation == nil {
		cellActivation = Tanh
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
			return nil, errors.Wrap(err, "Can't add batch dimension to LSTM layer's input")
		}
		input = reshaped
	}
	if input.Dims() != 3 {
		return nil, fmt.Errorf("LSTM layer's input should have shape [sequence, input_features] or [sequence, batch, input_features], got %v", input.Shape())
	}
	sequenceLength := input.Shape()[0]
	batch := input.Shape()[1]

	// Zero-valued initial states unless custom ones are provided.
	// Names must be unique in scope of graph: Gorgonia hashes input nodes by type, shape and name only
	if hiddenState == nil {
		hiddenState = gorgonia.NewMatrix(input.Graph(), input.Dtype(), gorgonia.WithShape(batch, hiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName(fmt.Sprintf("lstm_%d_initial_hidden", input.ID())))
	}
	if cellState == nil {
		cellState = gorgonia.NewMatrix(input.Graph(), input.Dtype(), gorgonia.WithShape(batch, hiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName(fmt.Sprintf("lstm_%d_initial_cell", input.ID())))
	}

	outputs := make([]*gorgonia.Node, sequenceLength)
	for t := 0; t < sequenceLength; t++ {
		// x_t of shape [batch, input_features]
		xt, err := gorgonia.Slice(input, gorgonia.S(t), nil, nil)
		if err != nil {
			return nil, errors.Wrapf(err, "Can't slice time step %d of LSTM layer's input of shape %v", t, input.Shape())
		}
		inputProjection, err := gorgonia.Mul(xt, layer.InputWeightNode)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiply time step input and input weights of LSTM layer")
		}
		hiddenProjection, err := gorgonia.Mul(hiddenState, layer.HiddenWeightNode)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiply previous hidden state and hidden weights of LSTM layer")
		}
		gates, err := gorgonia.Add(inputProjection, hiddenProjection)
		if err != nil {
			return nil, errors.Wrap(err, "Can't sum input and hidden projections of LSTM layer")
		}
		gates, err = addBias(gates, layer.BiasNode, batch)
		if err != nil {
			return nil, errors.Wrap(err, "Can't add bias to gates of LSTM layer")
		}
		// Gates are slices of columns: input, forget, cell candidate, output
		inputGate, err := sliceGate(gates, 0, hiddenSize, gatesActivation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't extract input gate of LSTM layer")
		}
		forgetGate, err := sliceGate(gates, 1, hiddenSize, gatesActivation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't extract forget gate of LSTM layer")
		}
		cellCandidate, err := sliceGate(gates, 2, hiddenSize, cellActivation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't extract cell candidate of LSTM layer")
		}
		outputGate, err := sliceGate(gates, 3, hiddenSize, gatesActivation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't extract output gate of LSTM layer")
		}
		// c_t = forget ⊙ c_prev + input ⊙ candidate
		preservedCell, err := gorgonia.HadamardProd(forgetGate, cellState)
		if err != nil {
			return nil, errors.Wrap(err, "Can't apply forget gate to previous cell state of LSTM layer")
		}
		incomingCell, err := gorgonia.HadamardProd(inputGate, cellCandidate)
		if err != nil {
			return nil, errors.Wrap(err, "Can't apply input gate to cell candidate of LSTM layer")
		}
		cellState, err = gorgonia.Add(preservedCell, incomingCell)
		if err != nil {
			return nil, errors.Wrap(err, "Can't update cell state of LSTM layer")
		}
		// h_t = output ⊙ activation(c_t)
		cellStateActivated, err := cellActivation(cellState)
		if err != nil {
			return nil, errors.Wrap(err, "Can't apply activation function to cell state of LSTM layer")
		}
		hiddenState, err = gorgonia.HadamardProd(outputGate, cellStateActivated)
		if err != nil {
			return nil, errors.Wrap(err, "Can't update hidden state of LSTM layer")
		}
		outputs[t], err = gorgonia.Reshape(hiddenState, tensor.Shape{1, batch, hiddenSize})
		if err != nil {
			return nil, errors.Wrap(err, "Can't add time dimension to hidden state of LSTM layer")
		}
	}
	layer.finalHiddenNode = hiddenState
	layer.finalCellNode = cellState

	layerNonActivated, err := gorgonia.Concat(0, outputs...)
	if err != nil {
		return nil, errors.Wrap(err, "Can't concatenate hidden states of LSTM layer")
	}
	if squeezeOutput {
		layerNonActivated, err = gorgonia.Reshape(layerNonActivated, tensor.Shape{sequenceLength, hiddenSize})
		if err != nil {
			return nil, errors.Wrap(err, "Can't squeeze batch dimension of LSTM layer's output")
		}
	}
	return layerNonActivated, nil
}

// Activate LSTM layer does not imply activation of output: activation functions are applied inside of the cell
func (layer *LSTMLayer) Activate(input *gorgonia.Node) (*gorgonia.Node, error) {
	return input, nil
}

// Learnables Returns learnable nodes
func (layer *LSTMLayer) Learnables() gorgonia.Nodes {
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
func (layer *LSTMLayer) CloneTo(g *gorgonia.ExprGraph, nameSuffix string) (Layer, error) {
	if layer.InputWeightNode == nil {
		return nil, fmt.Errorf("LSTM layer has nil input weights node")
	}
	if layer.HiddenWeightNode == nil {
		return nil, fmt.Errorf("LSTM layer has nil hidden weights node")
	}
	return &LSTMLayer{
		InputWeightNode:     cloneLearnableTo(g, layer.InputWeightNode, nameSuffix),
		HiddenWeightNode:    cloneLearnableTo(g, layer.HiddenWeightNode, nameSuffix),
		BiasNode:            cloneLearnableTo(g, layer.BiasNode, nameSuffix),
		InitialHiddenNode:   cloneLearnableTo(g, layer.InitialHiddenNode, nameSuffix),
		InitialCellNode:     cloneLearnableTo(g, layer.InitialCellNode, nameSuffix),
		HiddenSize:          layer.HiddenSize,
		Activation:          layer.Activation,
		RecurrentActivation: layer.RecurrentActivation,
	}, nil
}
