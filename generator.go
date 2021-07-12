package gan_go

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

// GeneratorNet Abstraction for generator part of GAN
//
// Layers - simple sequence of layers
// out - alias to activated output of last layer
//
type GeneratorNet struct {
	private *Network
}

// Generator Constructor for GeneratorNet
func Generator(Layers ...*Layer) *GeneratorNet {
	return &GeneratorNet{private: &Network{
		Layers: Layers,
	}}
}

// Out Returns reference to output node
func (net *GeneratorNet) Out() *gorgonia.Node {
	return net.private.out
}

// Learnables Returns learnables nodes
func (net *GeneratorNet) Learnables() gorgonia.Nodes {
	return net.private.Learnables()
}

// Fwd Initializates feedforward for provided input
//
// input - Input node
// batchSize - batch size. If it's >= 2 then broadcast function will be applied
//
func (net *GeneratorNet) Fwd(input *gorgonia.Node, batchSize int) error {
	if err := net.private.Fwd(input, batchSize); err != nil {
		return errors.Wrap(err, "[Generator]")
	}
	return nil
}
