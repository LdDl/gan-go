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

	KernelHeight int
	KernelWidth  int
	Padding      []int
	Stride       []int
	Dilation     []int
	ReshapeDims  []int
	Probability  float64
}

type LayerType uint16

const (
	LayerLinear = LayerType(iota)
	LayerFlatten
	LayerConvolutional
	LayerMaxpool
	LayerReshape
	LayerDropout
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
// input - Input node
// batchSize - batch size. If it's >= 2 then broadcast function will be applied
//
func (layer *Layer) Fwd(input *gorgonia.Node, batchSize int) (*gorgonia.Node, error) {
	var err error
	layerNonActivated := &gorgonia.Node{}

	if layer.WeightNode == nil && !noWeightsAllowed(layer.Type) {
		return nil, fmt.Errorf("Layer's weights node is nil")
	}

	switch layer.Type {
	case LayerLinear:
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
		break
	case LayerConvolutional:
		layerNonActivated, err = gorgonia.Conv2d(input, layer.WeightNode, tensor.Shape{layer.KernelHeight, layer.KernelWidth}, layer.Padding, layer.Stride, layer.Dilation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't convolve[2D] input by kernel of layer")
		}
		break
	case LayerMaxpool:
		layerNonActivated, err = gorgonia.MaxPool2D(input, tensor.Shape{layer.KernelHeight, layer.KernelWidth}, layer.Padding, layer.Stride)
		if err != nil {
			return nil, errors.Wrap(err, "Can't maxpool[2D] input by kernel of layer")
		}
		break
	case LayerFlatten:
		layerNonActivated, err = gorgonia.Reshape(input, tensor.Shape{batchSize, input.Shape().TotalSize() / batchSize})
		if err != nil {
			return nil, errors.Wrap(err, "Can't flatten input of layer")
		}
		break
	case LayerReshape:
		layerNonActivated, err = gorgonia.Reshape(input, layer.ReshapeDims)
		if err != nil {
			return nil, errors.Wrap(err, "Can't reshape input of layer")
		}
		break
	case LayerDropout:
		// Help developers to not provide NoActivation for dropout layer
		layer.Activation = NoActivation
		if ok := checkF64ValueInRange(layer.Probability, 0.0, 1.0); !ok {
			return nil, fmt.Errorf("Dropout probability should lie in [0;1] for layer. Got %f", layer.Probability)
		}
		layerNonActivated, err = gorgonia.Dropout(input, layer.Probability)
		if err != nil {
			return nil, errors.Wrap(err, "Can't dilute input of layer")
		}
		break
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

	return layerNonActivated, nil
}
