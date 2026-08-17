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
// is defined on its own graph (where it is trained) and its structure is copied into the GAN's graph.
// Each copied node is created with gorgonia.WithValue(originalNode.Value()) which binds the SAME
// underlying tensor (same backing memory) to both nodes — it is NOT a deep copy.
// Since Gorgonia's solvers update weights in-place, every training step of the Discriminator
// on its own graph is immediately "visible" to the copied nodes in the GAN's graph.
// So there is no need to manually sync weights between the two graphs.
// During the Generator's training step the solver is given GeneratorLearnables() only,
// therefore the Discriminator's copies stay untouched (they act as constants w.r.t. the update).
//
type GAN struct {
	generatorPart     *GeneratorNet
	discriminatorPart *DiscriminatorNet

	modifiedDiscriminator *DiscriminatorNet

	out           *gorgonia.Node
	learnables    gorgonia.Nodes
	learnablesGen gorgonia.Nodes
}

func NewGAN(g *gorgonia.ExprGraph, definedGenerator *GeneratorNet, definedDiscriminator *DiscriminatorNet) (*GAN, error) {
	definedGAN := GAN{
		generatorPart:     definedGenerator,
		discriminatorPart: definedDiscriminator,
		modifiedDiscriminator: &DiscriminatorNet{private: &Network{
			Name:   "gan_discriminator",
			Layers: make([]*Layer, len(definedDiscriminator.private.Layers)),
		}},
		learnablesGen: definedGenerator.Learnables(),
		learnables:    definedGenerator.Learnables(),
	}
	// Discriminator part for GAN
	for i, l := range definedDiscriminator.private.Layers {
		definedGAN.modifiedDiscriminator.private.Layers[i] = &Layer{
			Activation: l.Activation,
			Type:       l.Type,
		}
		if l.Options != nil {
			definedGAN.modifiedDiscriminator.private.Layers[i].Options = &Options{
				KernelHeight:  l.Options.KernelHeight,
				KernelWidth:   l.Options.KernelWidth,
				Padding:       l.Options.Padding,
				Stride:        l.Options.Stride,
				Dilation:      l.Options.Dilation,
				ReshapeDims:   l.Options.ReshapeDims,
				Probability:   l.Options.Probability,
				EmbeddingSize: l.Options.EmbeddingSize,
				Axis:          l.Options.Axis,
			}
		}
		if l.WeightNode == nil && !noWeightsAllowed(l.Type) {
			return nil, fmt.Errorf("Discriminator's Layer %d has nil weight node", i)
		}
		if l.WeightNode != nil {
			// WithValue(...) below binds the very same tensor as the original node has.
			// Backing memory is shared, so in-place updates done by a solver on the
			// Discriminator's own graph automatically propagate here. See note above.
			definedGAN.modifiedDiscriminator.private.Layers[i].WeightNode = gorgonia.NewTensor(g, l.WeightNode.Dtype(), l.WeightNode.Dims(), gorgonia.WithShape(l.WeightNode.Shape()...), gorgonia.WithName(l.WeightNode.Name()+"_gan"), gorgonia.WithValue(l.WeightNode.Value()))
			definedGAN.learnables = append(definedGAN.learnables, definedGAN.modifiedDiscriminator.private.Layers[i].WeightNode)
		}
		if l.BiasNode != nil {
			definedGAN.modifiedDiscriminator.private.Layers[i].BiasNode = gorgonia.NewTensor(g, l.BiasNode.Dtype(), l.BiasNode.Dims(), gorgonia.WithShape(l.BiasNode.Shape()...), gorgonia.WithName(l.BiasNode.Name()+"_gan"), gorgonia.WithValue(l.BiasNode.Value()))
			definedGAN.learnables = append(definedGAN.learnables, definedGAN.modifiedDiscriminator.private.Layers[i].BiasNode)
		}

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
//
func (net *GAN) Fwd(batchSize int) error {
	if err := net.modifiedDiscriminator.Fwd(batchSize, net.generatorPart.Out()); err != nil {
		return errors.Wrap(err, "[GAN, Discriminator part]")
	}
	net.out = net.modifiedDiscriminator.private.out
	return nil
}
