package gan_go

import (
	"gorgonia.org/gorgonia"
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
