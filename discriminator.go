package gan_go

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

// DiscriminatorNet Abstraction for discriminator part of GAN. It's simple neural network actually.
//
// Layers - simple sequence of layers
// out - alias to activated output of last layer
//
type DiscriminatorNet struct {
	private *Network
}

// Discriminator Constructor for DiscriminatorNet
func Discriminator(Layers ...*Layer) *DiscriminatorNet {
	return &DiscriminatorNet{private: &Network{
		Name:   "discriminator",
		Layers: Layers,
	}}
}

// Out Returns reference to output node
func (net *DiscriminatorNet) Out() *gorgonia.Node {
	return net.private.out
}

// Learnables Returns learnables nodes
func (net *DiscriminatorNet) Learnables() gorgonia.Nodes {
	return net.private.Learnables()
}

// Fwd Initializates feedforward for provided input
//
// input - Input node
// batchSize - batch size. If it's >= 2 then broadcast function will be applied
//
func (net *DiscriminatorNet) Fwd(input *gorgonia.Node, batchSize int) error {
	if err := net.private.Fwd(input, batchSize); err != nil {
		return errors.Wrap(err, "[Discriminator]")
	}
	return nil
}
