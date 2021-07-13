package gan_go

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

type LossReduction uint16

const (
	LossReductionSum = LossReduction(iota)
	LossReductionMean
	LossReductionMean33
)

// MSELoss See ref. https://en.wikipedia.org/wiki/Mean_squared_error
// Default reduction is 'mean'
func MSELoss(a, b *gorgonia.Node, reduction ...LossReduction) (*gorgonia.Node, error) {
	var err error
	var sub *gorgonia.Node
	sub, err = gorgonia.Sub(a, b)
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
