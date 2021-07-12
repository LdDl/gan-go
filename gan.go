package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// GAN Simple implementation of GAN.
//
// generatorPart - reference to Generator
// discriminatorPart - reference to Discriminator
// modifiedDiscriminator - copy of structure of Discriminator which learnables would be ignored during the training process
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
		// modifiedDiscriminator: make([]*Layer, len(definedDiscriminator.private.Layers)),
		modifiedDiscriminator: &DiscriminatorNet{private: &Network{
			Name:   "gan_discriminator",
			Layers: make([]*Layer, len(definedDiscriminator.private.Layers)),
		}},
		learnablesGen: definedGenerator.Learnables(),
		learnables:    append(definedGenerator.Learnables()),
	}
	// Discriminator part for GAN
	for i, l := range definedDiscriminator.private.Layers {
		definedGAN.modifiedDiscriminator.private.Layers[i] = &Layer{
			Activation:   l.Activation,
			Type:         l.Type,
			KernelHeight: l.KernelHeight,
			KernelWidth:  l.KernelWidth,
			Padding:      l.Padding,
			Stride:       l.Stride,
			Dilation:     l.Dilation,
			ReshapeDims:  l.ReshapeDims,
		}
		if l.WeightNode == nil && !noWeightsAllowed(l.Type) {
			return nil, fmt.Errorf("Discriminator's Layer %d has nil weight node", i)
		}
		if l.WeightNode != nil {
			definedGAN.modifiedDiscriminator.private.Layers[i].WeightNode = gorgonia.NewTensor(g, gorgonia.Float64, l.WeightNode.Dims(), gorgonia.WithShape(l.WeightNode.Shape()...), gorgonia.WithName(l.WeightNode.Name()+"_gan"), gorgonia.WithValue(l.WeightNode.Value()))
			definedGAN.learnables = append(definedGAN.learnables, definedGAN.modifiedDiscriminator.private.Layers[i].WeightNode)
		}
		if l.BiasNode != nil {
			definedGAN.modifiedDiscriminator.private.Layers[i].BiasNode = gorgonia.NewTensor(g, gorgonia.Float64, l.BiasNode.Dims(), gorgonia.WithShape(l.BiasNode.Shape()...), gorgonia.WithName(l.BiasNode.Name()+"_gan"), gorgonia.WithValue(l.BiasNode.Value()))
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
	var err error

	if len(net.modifiedDiscriminator.private.Layers) == 0 {
		return fmt.Errorf("GAN must have one layer in Discriminator part atleast")
	}
	if net.modifiedDiscriminator.private.Layers[0] == nil {
		return fmt.Errorf("GAN layer #0 [Discriminator part] is nil")
	}
	if net.modifiedDiscriminator.private.Layers[0].WeightNode == nil && !noWeightsAllowed(net.modifiedDiscriminator.private.Layers[0].Type) {
		return fmt.Errorf("GAN layer #0 WeightNode [Discriminator part] is nil")
	}

	firstLayerNonActivated := &gorgonia.Node{}
	switch net.modifiedDiscriminator.private.Layers[0].Type {
	case LayerLinear:
		tOp, err := gorgonia.Transpose(net.modifiedDiscriminator.private.Layers[0].WeightNode)
		if err != nil {
			return errors.Wrap(err, "Can't transpose weights of GAN's layer #0 [Discriminator part]")
		}
		if batchSize < 2 {
			firstLayerNonActivated, err = gorgonia.Mul(net.generatorPart.Out(), tOp)
			if err != nil {
				return errors.Wrap(err, "Can't multiply input and weights of GAN's layer #0 [Discriminator part]")
			}
		} else {
			firstLayerNonActivated, err = gorgonia.BatchedMatMul(net.generatorPart.Out(), tOp)
			if err != nil {
				return errors.Wrap(err, "Can't multiply input and weights of GAN's layer #0 [Discriminator part]")
			}
		}
		break
	case LayerConvolutional:
		firstLayerNonActivated, err = gorgonia.Conv2d(net.generatorPart.Out(), net.modifiedDiscriminator.private.Layers[0].WeightNode, tensor.Shape{net.modifiedDiscriminator.private.Layers[0].KernelHeight, net.modifiedDiscriminator.private.Layers[0].KernelWidth}, net.modifiedDiscriminator.private.Layers[0].Padding, net.modifiedDiscriminator.private.Layers[0].Stride, net.modifiedDiscriminator.private.Layers[0].Dilation)
		if err != nil {
			return errors.Wrap(err, "Can't convolve[2D] input by kernel of GAN's layer #0 [Discriminator part]")
		}
		break
	case LayerMaxpool:
		firstLayerNonActivated, err = gorgonia.MaxPool2D(net.generatorPart.Out(), tensor.Shape{net.modifiedDiscriminator.private.Layers[0].KernelHeight, net.modifiedDiscriminator.private.Layers[0].KernelWidth}, net.modifiedDiscriminator.private.Layers[0].Padding, net.modifiedDiscriminator.private.Layers[0].Stride)
		if err != nil {
			return errors.Wrap(err, "Can't maxpool[2D] input by kernel of GAN's layer #0 [Discriminator part]")
		}
		break
	case LayerFlatten:
		firstLayerNonActivated, err = gorgonia.Reshape(net.generatorPart.Out(), tensor.Shape{batchSize, net.generatorPart.Out().Shape().TotalSize() / batchSize})
		if err != nil {
			return errors.Wrap(err, "Can't flatten input of GAN's layer #0 [Discriminator part]")
		}
		break
	case LayerReshape:
		firstLayerNonActivated, err = gorgonia.Reshape(net.generatorPart.Out(), net.modifiedDiscriminator.private.Layers[0].ReshapeDims)
		if err != nil {
			return errors.Wrap(err, "Can't reshape input of GAN's layer #0 [Discriminator part]")
		}
		break
	default:
		return fmt.Errorf("Layer #0's type '%d' (uint16) is not handled [GAN]", net.modifiedDiscriminator.private.Layers[0].Type)
	}

	gorgonia.WithName("gan_discriminator_0")(firstLayerNonActivated)
	if net.modifiedDiscriminator.private.Layers[0].BiasNode != nil {
		if batchSize < 2 {
			firstLayerNonActivated, err = gorgonia.Add(firstLayerNonActivated, net.modifiedDiscriminator.private.Layers[0].BiasNode)
			if err != nil {
				return errors.Wrap(err, "Can't add bias to non-activated output of GAN's layer #0 [Discriminator part]")
			}
		} else {
			firstLayerNonActivated, err = gorgonia.BroadcastAdd(firstLayerNonActivated, net.modifiedDiscriminator.private.Layers[0].BiasNode, nil, []byte{0})
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't add [in broadcast term with batch_size = %d] bias to non-activated output of GAN's layer #0 [Discriminator part]", batchSize))
			}
		}
	}
	firstLayerActivated, err := net.modifiedDiscriminator.private.Layers[0].Activation(firstLayerNonActivated)
	if err != nil {
		return errors.Wrap(err, "Can't apply activation function to non-activated output of GAN's layer #0 [Discriminator part]")
	}
	gorgonia.WithName("gan_discriminator_activated_0")(firstLayerActivated)
	lastActivatedLayer := firstLayerActivated
	if len(net.modifiedDiscriminator.private.Layers) == 1 {
		net.out = lastActivatedLayer
	}
	for i := 1; i < len(net.modifiedDiscriminator.private.Layers); i++ {
		if net.modifiedDiscriminator.private.Layers[i] == nil {
			return fmt.Errorf("GAN layer #%d [Discriminator part] is nil", i)
		}
		if net.modifiedDiscriminator.private.Layers[i].WeightNode == nil && !noWeightsAllowed(net.modifiedDiscriminator.private.Layers[i].Type) {
			return fmt.Errorf("GAN layer's #%d WeightNode [Discriminator part] is nil", i)
		}
		layerNonActivated := &gorgonia.Node{}
		switch net.modifiedDiscriminator.private.Layers[i].Type {
		case LayerLinear:
			tOp, err := gorgonia.Transpose(net.modifiedDiscriminator.private.Layers[i].WeightNode)
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't transpose weights of GAN's layer #%d [Discriminator part]", i))
			}
			if batchSize < 2 {
				layerNonActivated, err = gorgonia.Mul(lastActivatedLayer, tOp)
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("Can't multiply input and weights of GAN's layer #%d [Discriminator part]", i))
				}
			} else {
				layerNonActivated, err = gorgonia.BatchedMatMul(lastActivatedLayer, tOp)
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("Can't multiply input and weights of GAN's layer #%d [Discriminator part]", i))
				}
			}
			break
		case LayerConvolutional:
			layerNonActivated, err = gorgonia.Conv2d(lastActivatedLayer, net.modifiedDiscriminator.private.Layers[i].WeightNode, tensor.Shape{net.modifiedDiscriminator.private.Layers[i].KernelHeight, net.modifiedDiscriminator.private.Layers[i].KernelWidth}, net.modifiedDiscriminator.private.Layers[i].Padding, net.modifiedDiscriminator.private.Layers[i].Stride, net.modifiedDiscriminator.private.Layers[i].Dilation)
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't convolve[2D] input by kernel of GAN's layer #%d [Discriminator part]", i))
			}
			break
		case LayerMaxpool:
			layerNonActivated, err = gorgonia.MaxPool2D(lastActivatedLayer, tensor.Shape{net.modifiedDiscriminator.private.Layers[i].KernelHeight, net.modifiedDiscriminator.private.Layers[i].KernelWidth}, net.modifiedDiscriminator.private.Layers[i].Padding, net.modifiedDiscriminator.private.Layers[i].Stride)
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't maxpool[2D] input by kernel of GAN's layer #%d [Discriminator part]", i))
			}
			break
		case LayerFlatten:
			layerNonActivated, err = gorgonia.Reshape(lastActivatedLayer, tensor.Shape{batchSize, lastActivatedLayer.Shape().TotalSize() / batchSize})
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't flatten input of GAN's layer #%d [Discriminator part]", i))
			}
			break
		case LayerReshape:
			layerNonActivated, err = gorgonia.Reshape(lastActivatedLayer, net.modifiedDiscriminator.private.Layers[i].ReshapeDims)
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't reshape input of GAN's layer #%d [Discriminator part]", i))
			}
			break
		default:
			return fmt.Errorf("Layer #%d's type '%d' (uint16) is not handled [GAN]", i, net.modifiedDiscriminator.private.Layers[i].Type)
		}

		gorgonia.WithName(fmt.Sprintf("gan_discriminator_%d", i))(layerNonActivated)
		if net.modifiedDiscriminator.private.Layers[i].BiasNode != nil {
			if batchSize < 2 {
				layerNonActivated, err = gorgonia.Add(layerNonActivated, net.modifiedDiscriminator.private.Layers[i].BiasNode)
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("Can't add bias to non-activated output of GAN's layer #%d [Discriminator part]", i))
				}
			} else {
				layerNonActivated, err = gorgonia.BroadcastAdd(layerNonActivated, net.modifiedDiscriminator.private.Layers[i].BiasNode, nil, []byte{0})
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("Can't add bias [in broadcast term with batch_size = %d] to non-activated output of GAN's layer #%d [Discriminator part]", batchSize, i))
				}
			}
		}
		layerActivated, err := net.modifiedDiscriminator.private.Layers[i].Activation(layerNonActivated)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("Can't apply activation function to non-activated output of GAN's layer #%d [Discriminator part]", i))
		}
		gorgonia.WithName(fmt.Sprintf("gan_discriminator_%d", i))(layerActivated)
		lastActivatedLayer = layerActivated
		if i == len(net.modifiedDiscriminator.private.Layers)-1 {
			net.out = layerActivated
		}
	}
	return nil
}
