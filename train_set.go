package gan_go

import (
	"gorgonia.org/tensor"
)

type TrainSet struct {
	TrainData  *tensor.Dense
	TrainLabel *tensor.Dense
	DataLength int
}
