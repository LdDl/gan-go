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
// modifiedDiscriminator - copy of structure of Discriminator which learnables would be ingored during the training process
//
type GAN struct {
	generatorPart     *Generator
	discriminatorPart *Discriminator

	modifiedDiscriminator []*Layer

	out           *gorgonia.Node
	learnables    gorgonia.Nodes
	learnablesGen gorgonia.Nodes
}

func NewGAN(g *gorgonia.ExprGraph, definedGenerator *Generator, definedDiscriminator *Discriminator) (*GAN, error) {
	definedGAN := GAN{
		generatorPart:         definedGenerator,
		discriminatorPart:     definedDiscriminator,
		modifiedDiscriminator: make([]*Layer, len(definedDiscriminator.Layers)),
		learnablesGen:         definedGenerator.Learnables(),
		learnables:            append(definedGenerator.Learnables()),
	}
	// Discriminator part for GAN
	for i, l := range definedDiscriminator.Layers {
		definedGAN.modifiedDiscriminator[i] = &Layer{
			Activation: l.Activation,
		}
		if l.WeightNode == nil {
			return nil, fmt.Errorf("Discriminator's Layer %d has nil weight node", i)
		}
		definedGAN.modifiedDiscriminator[i].WeightNode = gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(l.WeightNode.Shape()...), gorgonia.WithName(l.WeightNode.Name()+"_gan"), gorgonia.WithValue(l.WeightNode.Value()))
		if l.BiasNode != nil {
			definedGAN.modifiedDiscriminator[i].BiasNode = gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(l.BiasNode.Shape()...), gorgonia.WithName(l.BiasNode.Name()+"_gan"), gorgonia.WithValue(l.BiasNode.Value()))
		}
		definedGAN.learnables = append(definedGAN.learnables, definedGAN.modifiedDiscriminator[i].WeightNode, definedGAN.modifiedDiscriminator[i].BiasNode)
	}
	return &definedGAN, nil
}

// Out Returns reference to output node
func (net *GAN) Out() *gorgonia.Node {
	return net.out
}

// GeneratorOut Returns reference to output node of generator part
func (net *GAN) GeneratorOut() *gorgonia.Node {
	return net.generatorPart.out
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
	if len(net.modifiedDiscriminator) == 0 {
		return fmt.Errorf("GAN must have one layer in Discriminator part atleast")
	}
	if net.modifiedDiscriminator[0] == nil {
		return fmt.Errorf("GAN layer #0 [Discriminator part] is nil")
	}
	if net.modifiedDiscriminator[0].WeightNode == nil {
		return fmt.Errorf("GAN layer #0 WeightNode [Discriminator part] is nil")
	}
	tOp, err := gorgonia.Transpose(net.modifiedDiscriminator[0].WeightNode)
	if err != nil {
		return errors.Wrap(err, "Can't transpose weights of GAN's layer #0 [Discriminator part]")
	}
	firstLayerNonActivated, err := gorgonia.Mul(net.generatorPart.Out(), tOp)
	if err != nil {
		return errors.Wrap(err, "Can't multiply input and weights of GAN's layer #0 [Discriminator part]")
	}
	gorgonia.WithName("gan_discriminator_0")(firstLayerNonActivated)
	if net.modifiedDiscriminator[0].BiasNode != nil {
		if batchSize < 2 {
			firstLayerNonActivated, err = gorgonia.Add(firstLayerNonActivated, net.modifiedDiscriminator[0].BiasNode)
			if err != nil {
				return errors.Wrap(err, "Can't add bias to non-activated output of GAN's layer #0 [Discriminator part]")
			}
		} else {
			firstLayerNonActivated, err = gorgonia.BroadcastAdd(firstLayerNonActivated, net.modifiedDiscriminator[0].BiasNode, nil, []byte{0})
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't add [in broadcast term with batch_size = %d] bias to non-activated output of GAN's layer #0 [Discriminator part]", batchSize))
			}
		}
	}
	firstLayerActivated, err := net.modifiedDiscriminator[0].Activation(firstLayerNonActivated)
	if err != nil {
		return errors.Wrap(err, "Can't apply activation function to non-activated output of GAN's layer #0 [Discriminator part]")
	}
	gorgonia.WithName("gan_discriminator_activated_0")(firstLayerActivated)
	lastActivatedLayer := firstLayerActivated
	for i := 1; i < len(net.modifiedDiscriminator); i++ {
		if net.modifiedDiscriminator[i] == nil {
			return fmt.Errorf("GAN layer #%d [Discriminator part] is nil", i)
		}
		if net.modifiedDiscriminator[i].WeightNode == nil {
			return fmt.Errorf("GAN layer's #%d WeightNode [Discriminator part] is nil", i)
		}
		tOp, err := gorgonia.Transpose(net.modifiedDiscriminator[i].WeightNode)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("Can't transpose weights of GAN's layer #%d [Discriminator part]", i))
		}
		layerNonActivated, err := gorgonia.Mul(lastActivatedLayer, tOp)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("Can't multiply input and weights of GAN's layer #%d [Discriminator part]", i))
		}
		gorgonia.WithName(fmt.Sprintf("gan_discriminator_%d", i))(layerNonActivated)
		if net.modifiedDiscriminator[0].BiasNode != nil {
			if batchSize < 2 {
				layerNonActivated, err = gorgonia.Add(layerNonActivated, net.modifiedDiscriminator[i].BiasNode)
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("Can't add bias to non-activated output of GAN's layer #%d [Discriminator part]", i))
				}
			} else {
				layerNonActivated, err = gorgonia.BroadcastAdd(layerNonActivated, net.modifiedDiscriminator[i].BiasNode, nil, []byte{0})
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("Can't add bias [in broadcast term with batch_size = %d] to non-activated output of GAN's layer #%d [Discriminator part]", batchSize, i))
				}
			}
		}
		layerActivated, err := net.modifiedDiscriminator[i].Activation(layerNonActivated)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("Can't apply activation function to non-activated output of GAN's layer #%d [Discriminator part]", i))
		}
		gorgonia.WithName(fmt.Sprintf("gan_discriminator_%d", i))(layerActivated)
		lastActivatedLayer = layerActivated
		if i == len(net.modifiedDiscriminator)-1 {
			net.out = layerActivated
		}
	}
	return nil
}
