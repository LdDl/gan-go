package gan_go

import (
	"gorgonia.org/gorgonia"
)

// ActivationFunc Just an alias to Gorgonia'a api_gen.go - https://github.com/gorgonia/gorgonia/blob/master/api_gen.go#L1
type ActivationFunc func(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)

func NoActivation(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error) { return a, nil }
func Abs(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)          { return gorgonia.Abs(a) }
func Sign(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)         { return gorgonia.Sign(a) }
func Ceil(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)         { return gorgonia.Ceil(a) }
func Floor(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)        { return gorgonia.Floor(a) }
func Sin(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)          { return gorgonia.Sin(a) }
func Cos(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)          { return gorgonia.Cos(a) }
func Exp(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)          { return gorgonia.Exp(a) }
func Log(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)          { return gorgonia.Log(a) }
func Log2(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)         { return gorgonia.Log2(a) }
func Neg(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)          { return gorgonia.Neg(a) }
func Square(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)       { return gorgonia.Square(a) }
func Sqrt(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)         { return gorgonia.Sqrt(a) }
func Inverse(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)      { return gorgonia.Inverse(a) }
func InverseSqrt(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error) {
	return gorgonia.InverseSqrt(a)
}
func Cube(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)     { return gorgonia.Cube(a) }
func Tanh(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)     { return gorgonia.Tanh(a) }
func Sigmoid(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)  { return gorgonia.Sigmoid(a) }
func Log1p(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)    { return gorgonia.Log1p(a) }
func Expm1(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)    { return gorgonia.Expm1(a) }
func Softplus(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error) { return gorgonia.Softplus(a) }
func Rectify(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error)  { return gorgonia.Rectify(a) }
func Softmax(a *gorgonia.Node, opts ...Options) (*gorgonia.Node, error) {
	for i := range opts {
		// Check if axis option is provided
		// First i-th option with provided field 'Axis' would be considered for use.
		if len(opts[i].Axis) > 0 {
			return gorgonia.SoftMax(a, opts[i].Axis...)
		}
	}
	return gorgonia.SoftMax(a)
}

// Options Struct for holding options for certain activation functions.
type Options struct {
	Axis []int
}
