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
}

type LayerType uint16

const (
	LayerLinear = LayerType(iota)
	LayerFlatten
	LayerConvolutional
	LayerMaxpool
	LayerReshape
)

var (
	allowedNoWeights = []LayerType{LayerMaxpool, LayerFlatten, LayerReshape}
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
