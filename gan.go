package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

// GAN Simple implementation of GAN.
//
// generatorPart - reference to Generator
// discriminatorPart - reference to Discriminator
// modifiedDiscriminator - copy of structure of Discriminator which learnables would be ignored during the training process
//
// Note on the weights of modifiedDiscriminator (the "shared memory" trick):
// Gorgonia does not provide a way to "freeze" a subset of learnables, so the Discriminator
// is defined on its own graph (where it is trained) and its structure is copied into the GAN's graph
// via CloneTo(...) method of Layer interface.
// Each copied learnable node is created with gorgonia.WithValue(originalNode.Value()) which binds the SAME
// underlying tensor (same backing memory) to both nodes — it is NOT a deep copy.
// Since Gorgonia's solvers update weights in-place, every training step of the Discriminator
// on its own graph is immediately "visible" to the copied nodes in the GAN's graph.
// So there is no need to manually sync weights between the two graphs.
// During the Generator's training step the solver is given GeneratorLearnables() only,
// therefore the Discriminator's copies stay untouched (they act as constants w.r.t. the update).
type GAN struct {
	generatorPart     *GeneratorNet
	discriminatorPart *DiscriminatorNet

	modifiedDiscriminator *DiscriminatorNet

	out           *gorgonia.Node
	learnables    gorgonia.Nodes
	learnablesGen gorgonia.Nodes
}

// NewGAN Constructor for GAN.
//
// g - graph where Generator is defined (GAN's copy of Discriminator will be defined on the same graph)
// definedGenerator - reference to Generator
// definedDiscriminator - reference to Discriminator (could be defined on any graph)
func NewGAN(g *gorgonia.ExprGraph, definedGenerator *GeneratorNet, definedDiscriminator *DiscriminatorNet) (*GAN, error) {
	definedGAN := GAN{
		generatorPart:     definedGenerator,
		discriminatorPart: definedDiscriminator,
		modifiedDiscriminator: &DiscriminatorNet{private: &Network{
			Name:   "gan_discriminator",
			Layers: make([]Layer, len(definedDiscriminator.private.Layers)),
		}},
		learnablesGen: definedGenerator.Learnables(),
		learnables:    definedGenerator.Learnables(),
	}
	// Discriminator part for GAN
	for i, l := range definedDiscriminator.private.Layers {
		if l == nil {
			return nil, fmt.Errorf("Discriminator's Layer %d is nil", i)
		}
		clonedLayer, err := l.CloneTo(g, "_gan")
		if err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("Can't clone Discriminator's Layer %d onto GAN's graph", i))
		}
		definedGAN.modifiedDiscriminator.private.Layers[i] = clonedLayer
		definedGAN.learnables = append(definedGAN.learnables, clonedLayer.Learnables()...)
	}
	return &definedGAN, nil
}

// Out Returns reference to output node
func (net *GAN) Out() *gorgonia.Node {
	return net.out
}

// GeneratorOut Returns reference to output node of generator part
func (net *GAN) GeneratorOut() *gorgonia.Node {
	return net.generatorPart.Out()
}

// Learnables Returns learnables nodes
func (net *GAN) Learnables() gorgonia.Nodes {
	return net.learnables
}

// Learnables Returns learnables nodes of generator part
func (net *GAN) GeneratorLearnables() gorgonia.Nodes {
	return net.learnablesGen
}

// Fwd Initializates feedforward for provided input for disciminator part of GAN
//
// batchSize - batch size. If it's >= 2 then broadcast function will be applied
// Note: input node is not needed since input for Discriminator is just Generator's output
func (net *GAN) Fwd(batchSize int) error {
	if err := net.modifiedDiscriminator.Fwd(batchSize, net.generatorPart.Out()); err != nil {
		return errors.Wrap(err, "[GAN, Discriminator part]")
	}
	net.out = net.modifiedDiscriminator.private.out
	return nil
}
