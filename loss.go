package gan_go

import (
	"fmt"

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
		return nil, err
	}
	sqr, err := gorgonia.Square(sub)
	if err != nil {
		return nil, err
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
		return nil, err
	}
	neg, err := gorgonia.Neg(log)
	if err != nil {
		return nil, err
	}
	hprod, err := gorgonia.HadamardProd(neg, b)
	if err != nil {
		return nil, err
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
