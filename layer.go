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
}

// ActivationFunc Just an alias to Gorgonia'a api_gen.go - https://github.com/gorgonia/gorgonia/blob/master/api_gen.go#L1
// Note: you can implement function by yourself but here is set of suitable functions (so you don't need to make your own Sigmoid e.g.):
// func Abs(a *Node) (*Node, error)
// func Sign(a *Node) (*Node, error)
// func Ceil(a *Node) (*Node, error)
// func Floor(a *Node) (*Node, error)
// func Sin(a *Node) (*Node, error)
// func Cos(a *Node) (*Node, error)
// func Exp(a *Node) (*Node, error)
// func Log(a *Node) (*Node, error)
// func Log2(a *Node) (*Node, error)
// func Neg(a *Node) (*Node, error)
// func Square(a *Node) (*Node, error)
// func Sqrt(a *Node) (*Node, error)
// func Inverse(a *Node) (*Node, error)
// func InverseSqrt(a *Node) (*Node, error)
// func Cube(a *Node) (*Node, error)
// func Tanh(a *Node) (*Node, error)
// func Sigmoid(a *Node) (*Node, error)
// func Log1p(a *Node) (*Node, error)
// func Expm1(a *Node) (*Node, error)
// func Softplus(a *Node) (*Node, error)
type ActivationFunc func(a *gorgonia.Node) (*gorgonia.Node, error)

func NoActivation(a *gorgonia.Node) (*gorgonia.Node, error) {
	return a, nil
}

type LayerType uint16

const (
	LayerLinear = LayerType(iota)
)
