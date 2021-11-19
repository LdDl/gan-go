package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Layer Just an alias to Weight+Bias+ActivationFunction combo
type Layer struct {
	WeightNode *gorgonia.Node
	BiasNode   *gorgonia.Node
	Activation ActivationFunc
	Type       LayerType

	outputNonActivatedNode *gorgonia.Node
	outputActivatedNode    *gorgonia.Node
	extraNodes             []*gorgonia.Node
	Options                *Options
}

// Options Struct for holding options for certain activation functions.
// LayerOptions Struct for holding options for different layers' types

type Options struct {
	// Used in layers: [Conv2D, Maxpool2D]
	KernelHeight int
	// Used in layers: [Conv2D, Maxpool2D]
	KernelWidth int
	// Used in layers: [Conv2D, Maxpool2D]
	Padding []int
	// Used in layers: [Conv2D, Maxpool2D]
	Stride []int
	// Used in layers: [Conv2D, Maxpool2D]
	Dilation []int
	// Used in layers: [Reshape]
	ReshapeDims []int
	// Used in layers: [Dropout]
	Probability float64
	// Used in layers: [Embedding]
	EmbeddingSize int
	// Not used in layers directly. But used for defining parametetrs for certain activation functions (e.g. Softmax)
	Axis []int
}

type LayerType uint16

const (
	LayerLinear = LayerType(iota)
	LayerFlatten
	LayerConvolutional
	LayerMaxpool
	LayerReshape
	LayerDropout
	LayerEmbedding
)

var (
	allowedNoWeights = []LayerType{LayerMaxpool, LayerFlatten, LayerReshape, LayerDropout}
)

func noWeightsAllowed(checkType LayerType) bool {
	return checkLayerType(checkType, allowedNoWeights...)
}

func checkLayerType(checkType LayerType, t ...LayerType) bool {
	for _, typeOf := range t {
		if checkType == typeOf {
			return true
		}
	}
	return false
}

func checkF64ValueInRange(input, min, max float64) bool {
	if input > max && input < min {
		return false
	}
	return true
}

// Fwd Initializates feedforward for provided input
//
// inputs - Input node (or nodes)
// batchSize - batch size. If it's >= 2 then broadcast function will be applied
//
func (layer *Layer) Fwd(batchSize int, inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	var err error
	layerNonActivated := &gorgonia.Node{}

	if len(inputs) == 0 {
		return nil, fmt.Errorf("There are no input nodes for layer")
	}

	if layer.WeightNode == nil && !noWeightsAllowed(layer.Type) {
		return nil, fmt.Errorf("Layer's weights node is nil")
	}
// @todo: I guess it's better to have interfaces or personalized methods
	switch layer.Type {
	case LayerLinear:
		if len(inputs) > 1 {
			return nil, fmt.Errorf("Layer's type '%d'can handle only 1 input node, got %d", layer.Type, len(inputs))
		}
		input := inputs[0]
		tOp, err := gorgonia.Transpose(layer.WeightNode)
		if err != nil {
			return nil, errors.Wrap(err, "Can't transpose weights of layer")
		}
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
	case LayerConvolutional:
		if len(inputs) > 1 {
			return nil, fmt.Errorf("Layer's type '%d'can handle only 1 input node, got %d", layer.Type, len(inputs))
		}
		input := inputs[0]
		if layer.Options == nil {
			return nil, fmt.Errorf("Options haven't been provided for layer")
		}
		layerNonActivated, err = gorgonia.Conv2d(input, layer.WeightNode, tensor.Shape{layer.Options.KernelHeight, layer.Options.KernelWidth}, layer.Options.Padding, layer.Options.Stride, layer.Options.Dilation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't convolve[2D] input by kernel of layer")
		}
	case LayerMaxpool:
		if len(inputs) > 1 {
			return nil, fmt.Errorf("Layer's type '%d'can handle only 1 input node, got %d", layer.Type, len(inputs))
		}
		input := inputs[0]
		if layer.Options == nil {
			return nil, fmt.Errorf("Options haven't been provided for layer")
		}
		layerNonActivated, err = gorgonia.MaxPool2D(input, tensor.Shape{layer.Options.KernelHeight, layer.Options.KernelWidth}, layer.Options.Padding, layer.Options.Stride)
		if err != nil {
			return nil, errors.Wrap(err, "Can't maxpool[2D] input by kernel of layer")
		}
	case LayerFlatten:
		if len(inputs) > 1 {
			return nil, fmt.Errorf("Layer's type '%d'can handle only 1 input node, got %d", layer.Type, len(inputs))
		}
		input := inputs[0]
		// Help developers to not provide NoActivation for flatten layer
		layer.Activation = NoActivation
		layerNonActivated, err = gorgonia.Reshape(input, tensor.Shape{batchSize, input.Shape().TotalSize() / batchSize})
		if err != nil {
			return nil, errors.Wrap(err, "Can't flatten input of layer")
		}
	case LayerReshape:
		if len(inputs) > 1 {
			return nil, fmt.Errorf("Layer's type '%d'can handle only 1 input node, got %d", layer.Type, len(inputs))
		}
		input := inputs[0]
		// Help developers to not provide NoActivation for reshaping layer
		layer.Activation = NoActivation
		layerNonActivated, err = gorgonia.Reshape(input, layer.Options.ReshapeDims)
		if err != nil {
			return nil, errors.Wrap(err, "Can't reshape input of layer")
		}
	case LayerDropout:
		if len(inputs) > 1 {
			return nil, fmt.Errorf("Layer's type '%d'can handle only 1 input node, got %d", layer.Type, len(inputs))
		}
		input := inputs[0]
		// Help developers to not provide NoActivation for dropout layer
		layer.Activation = NoActivation
		if ok := checkF64ValueInRange(layer.Options.Probability, 0.0, 1.0); !ok {
			return nil, fmt.Errorf("Dropout probability should lie in [0;1] for layer. Got %f", layer.Options.Probability)
		}
		layerNonActivated, err = gorgonia.Dropout(input, layer.Options.Probability)
		if err != nil {
			return nil, errors.Wrap(err, "Can't dilute input of layer")
		}
	case LayerEmbedding:
		if len(inputs) > 1 {
			return nil, fmt.Errorf("Layer's type '%d'can handle only 1 input node, got %d", layer.Type, len(inputs))
		}
		input := inputs[0]
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
		// Help developers to not provide NoActivation for embedding layer
		layer.Activation = NoActivation
		layerNonActivated, err = gorgonia.Reshape(tmpEmbedding, append(input.Shape(), layer.Options.EmbeddingSize))
		if err != nil {
			return nil, errors.Wrap(err, "Can't embedd input of layer")
		}
	default:
		return nil, fmt.Errorf("Layer's type '%d' (uint16) is not handled", layer.Type)
	}

	// Bias part
	if layer.BiasNode != nil {
		if batchSize < 2 {
			layerNonActivated, err = gorgonia.Add(layerNonActivated, layer.BiasNode)
			if err != nil {
				return nil, errors.Wrap(err, "Can't add bias to non-activated output of a layer")
			}
		} else {
			layerNonActivated, err = gorgonia.BroadcastAdd(layerNonActivated, layer.BiasNode, nil, []byte{0})
			if err != nil {
				return nil, errors.Wrap(err, fmt.Sprintf("Can't add [in broadcast term with batch_size = %d] bias to non-activated output of a layer", batchSize))
			}
		}
	}
	layer.outputNonActivatedNode = layerNonActivated
	return layer.outputNonActivatedNode, nil
}
