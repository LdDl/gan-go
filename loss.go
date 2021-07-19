package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

type LossReduction uint16

const (
	LossReductionSum = LossReduction(iota)
	LossReductionMean
)

// MSELoss See ref. https://en.wikipedia.org/wiki/Mean_squared_error
// Default reduction is 'mean'
func MSELoss(a, b *gorgonia.Node, reduction ...LossReduction) (*gorgonia.Node, error) {
	sub, err := gorgonia.Sub(a, b)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (A-B)")
	}
	sqr, err := gorgonia.Square(sub)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (x^2)")
	}
	reductionDefault := LossReductionMean
	if len(reduction) != 0 {
		reductionDefault = reduction[0]
	}
	switch reductionDefault {
	case LossReductionSum:
		return gorgonia.Sum(sqr)
	case LossReductionMean:
		return gorgonia.Mean(sqr)
	default:
		return nil, fmt.Errorf("Reduction type %d is not supported", reductionDefault)
	}
}

// CrossEntropyLoss See ref. https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression
// Default reduction is 'mean'
func CrossEntropyLoss(a, b *gorgonia.Node, reduction ...LossReduction) (*gorgonia.Node, error) {
	log, err := gorgonia.Log(a)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do log(A)")
	}
	neg, err := gorgonia.Neg(log)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do -1*x")
	}
	hprod, err := gorgonia.HadamardProd(neg, b)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (x.*B)")
	}
	reductionDefault := LossReductionMean
	if len(reduction) != 0 {
		reductionDefault = reduction[0]
	}
	switch reductionDefault {
	case LossReductionSum:
		return gorgonia.Sum(hprod)
	case LossReductionMean:
		return gorgonia.Mean(hprod)
	default:
		return nil, fmt.Errorf("Reduction type %d is not supported", reductionDefault)
	}
}

// BinaryCrossEntropyLoss See ref. https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression
// Pretty the same as CrossEntropyLoss. BUT for C=2, where C - number of classes
// In case of binary variation of cross entropy loss: sample could belong to 0 or 1 only.
// Default reduction is 'mean'
func BinaryCrossEntropyLoss(a, b *gorgonia.Node, reduction ...LossReduction) (*gorgonia.Node, error) {
	// Main part the same as cross entropy
	logMain, err := gorgonia.Log(a)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do log(A)")
	}
	negMain, err := gorgonia.Neg(logMain)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do -1*x")
	}

	hprodMain, err := gorgonia.HadamardProd(negMain, b)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (x.*B)")
	}

	// Here comes another part
	onesTensor := gorgonia.NewTensor(a.Graph(), a.Dtype(), a.Dims(), gorgonia.WithShape(a.Shape()...), gorgonia.WithInit(gorgonia.Ones()))
	logBin, err := gorgonia.Sub(onesTensor, a)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do log(1-A)")
	}
	negBin, err := gorgonia.Neg(logBin)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do -1*x")
	}
	preLogBin, err := gorgonia.Sub(onesTensor, b)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (1-B)")
	}
	hprodBin, err := gorgonia.HadamardProd(negBin, preLogBin)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (x.*B)")
	}
	hprod, err := gorgonia.Add(hprodMain, hprodBin)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (x+y)")
	}

	reductionDefault := LossReductionMean
	if len(reduction) != 0 {
		reductionDefault = reduction[0]
	}

	switch reductionDefault {
	case LossReductionSum:
		return gorgonia.Sum(hprod)
	case LossReductionMean:
		return gorgonia.Mean(hprod)
	default:
		return nil, fmt.Errorf("Reduction type %d is not supported", reductionDefault)
	}
}

// L1Loss See ref. https://en.wikipedia.org/wiki/Least_absolute_deviations
// Default reduction is 'mean'
func L1Loss(a, b *gorgonia.Node, reduction ...LossReduction) (*gorgonia.Node, error) {
	sub, err := gorgonia.Sub(a, b)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (A-B)")
	}
	abs, err := gorgonia.Abs(sub)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do |x|")
	}

	reductionDefault := LossReductionMean
	if len(reduction) != 0 {
		reductionDefault = reduction[0]
	}
	switch reductionDefault {
	case LossReductionSum:
		return gorgonia.Sum(abs)
	case LossReductionMean:
		return gorgonia.Mean(abs)
	default:
		return nil, fmt.Errorf("Reduction type %d is not supported", reductionDefault)
	}
}

// HuberLoss See ref. https://en.wikipedia.org/wiki/Huber_loss
// This is actually Pseudo Huber Loss - see ref. https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
// Delta value type should match Dtype of provided nodes. tensor.Float32 -> float32, tensor.Float64 -> float64 and etc.
// Default reduction is 'mean'
func HuberLoss(a, b *gorgonia.Node, delta interface{}, reduction ...LossReduction) (*gorgonia.Node, error) {
	deltaScalar := gorgonia.NewScalar(a.Graph(), a.Dtype(), gorgonia.WithValue(delta))
	sqrDelta := gorgonia.NewScalar(a.Graph(), a.Dtype(), gorgonia.WithValue(delta))
	oneScalar := gorgonia.NewScalar(a.Graph(), a.Dtype(), gorgonia.WithValue(1.0))

	sub, err := gorgonia.Sub(a, b)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (A-B)")
	}

	div, err := gorgonia.Div(sub, deltaScalar)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (X/delta)")
	}
	sqr, err := gorgonia.Square(div)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (x^2)")
	}
	addOneScalar, err := gorgonia.Add(oneScalar, sqr)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (1.+X)")
	}
	sqrt, err := gorgonia.Sqrt(addOneScalar)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do √x")
	}
	subOneScalar, err := gorgonia.Sub(sqrt, oneScalar)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (X.-1)")
	}
	matMul, err := gorgonia.Mul(sqrDelta, subOneScalar)
	if err != nil {
		return nil, errors.Wrap(err, "Can't do (y@x)")
	}
	reductionDefault := LossReductionMean
	if len(reduction) != 0 {
		reductionDefault = reduction[0]
	}
	switch reductionDefault {
	case LossReductionSum:
		return gorgonia.Sum(matMul)
	case LossReductionMean:
		return gorgonia.Mean(matMul)
	default:
		return nil, fmt.Errorf("Reduction type %d is not supported", reductionDefault)
	}
}
