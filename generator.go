package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Generator Abstraction for generator part of GAN
//
// Layers - simple sequence of layers
// out - alias to activated output of last layer
//
type Generator struct {
	Layers []*Layer
	out    *gorgonia.Node
}

// Out Returns reference to output node
func (net *Generator) Out() *gorgonia.Node {
	return net.out
}

// Learnables Returns learnables nodes
func (net *Generator) Learnables() gorgonia.Nodes {
	learnables := make(gorgonia.Nodes, 0, 2*len(net.Layers))
	for _, l := range net.Layers {
		if l != nil {
			if l.WeightNode != nil {
				learnables = append(learnables, l.WeightNode)
			}
			if l.BiasNode != nil {
				learnables = append(learnables, l.BiasNode)
			}
		}
	}
	return learnables
}

// Fwd Initializates feedforward for provided input
//
// input - Input node
// batchSize - batch size. If it's >= 2 then broadcast function will be applied
//
func (net *Generator) Fwd(input *gorgonia.Node, batchSize int) error {
	var err error

	if len(net.Layers) == 0 {
		return fmt.Errorf("Generator must have one layer atleast")
	}
	if net.Layers[0] == nil {
		return fmt.Errorf("Generator's layer #0 is nil")
	}
	if net.Layers[0].WeightNode == nil && !noWeightsAllowed(net.Layers[0].Type) {
		return fmt.Errorf("Generator's layer #0 WeightNode is nil")
	}

	firstLayerNonActivated := &gorgonia.Node{}
	switch net.Layers[0].Type {
	case LayerLinear:
		tOp, err := gorgonia.Transpose(net.Layers[0].WeightNode)
		if err != nil {
			return errors.Wrap(err, "Can't transpose weights of Generator's layer #0")
		}
		if batchSize < 2 {
			firstLayerNonActivated, err = gorgonia.Mul(input, tOp)
			if err != nil {
				return errors.Wrap(err, "Can't multiply input and weights of Generator's layer #0")
			}
		} else {
			firstLayerNonActivated, err = gorgonia.BatchedMatMul(input, tOp)
			if err != nil {
				return errors.Wrap(err, "Can't multiply input and weights of Generator's layer #0")
			}
		}
		break
	case LayerConvolutional:
		firstLayerNonActivated, err = gorgonia.Conv2d(input, net.Layers[0].WeightNode, tensor.Shape{net.Layers[0].KernelHeight, net.Layers[0].KernelWidth}, net.Layers[0].Padding, net.Layers[0].Stride, net.Layers[0].Dilation)
		if err != nil {
			return errors.Wrap(err, "Can't convolve[2D] input by kernel of Generator's layer #0")
		}
		break
	case LayerMaxpool:
		firstLayerNonActivated, err = gorgonia.MaxPool2D(input, tensor.Shape{net.Layers[0].KernelHeight, net.Layers[0].KernelWidth}, net.Layers[0].Padding, net.Layers[0].Stride)
		if err != nil {
			return errors.Wrap(err, "Can't maxpool[2D] input by kernel of Generator's layer #0")
		}
		break
	case LayerFlatten:
		firstLayerNonActivated, err = gorgonia.Reshape(input, tensor.Shape{batchSize, input.Shape().TotalSize() / batchSize})
		if err != nil {
			return errors.Wrap(err, "Can't flatten input of Generator's layer #0")
		}
		break
	case LayerReshape:
		firstLayerNonActivated, err = gorgonia.Reshape(input, net.Layers[0].ReshapeDims)
		if err != nil {
			return errors.Wrap(err, "Can't reshape input of Generator's layer #0")
		}
		break
	default:
		return fmt.Errorf("Layer #0's type '%d' (uint16) is not handled [Generator]", net.Layers[0].Type)
	}

	gorgonia.WithName("generator_0")(firstLayerNonActivated)
	if net.Layers[0].BiasNode != nil {
		if batchSize < 2 {
			firstLayerNonActivated, err = gorgonia.Add(firstLayerNonActivated, net.Layers[0].BiasNode)
			if err != nil {
				return errors.Wrap(err, "Can't add bias to non-activated output of Generator's layer #0")
			}
		} else {
			firstLayerNonActivated, err = gorgonia.BroadcastAdd(firstLayerNonActivated, net.Layers[0].BiasNode, nil, []byte{0})
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't add [in broadcast term with batch_size = %d] bias to non-activated output of Generator's layer #0", batchSize))
			}
		}
	}
	firstLayerActivated, err := net.Layers[0].Activation(firstLayerNonActivated)
	if err != nil {
		return errors.Wrap(err, "Can't apply activation function to non-activated output of Generator's layer #0")
	}
	gorgonia.WithName("generator_activated_0")(firstLayerActivated)
	lastActivatedLayer := firstLayerActivated
	if len(net.Layers) == 1 {
		net.out = lastActivatedLayer
	}
	for i := 1; i < len(net.Layers); i++ {
		if net.Layers[i] == nil {
			return fmt.Errorf("Generator's layer #%d is nil", i)
		}
		if net.Layers[i].WeightNode == nil && !noWeightsAllowed(net.Layers[i].Type) {
			return fmt.Errorf("Generator's layer's #%d WeightNode is nil", i)
		}

		layerNonActivated := &gorgonia.Node{}
		switch net.Layers[i].Type {
		case LayerLinear:
			tOp, err := gorgonia.Transpose(net.Layers[i].WeightNode)
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't transpose weights of Generator's layer #%d", i))
			}
			if batchSize < 2 {
				layerNonActivated, err = gorgonia.Mul(lastActivatedLayer, tOp)
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("Can't multiply input and weights of Generator's layer #%d", i))
				}
			} else {
				layerNonActivated, err = gorgonia.BatchedMatMul(lastActivatedLayer, tOp)
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("Can't multiply input and weights of Generator's layer #%d", i))
				}
			}
			break
		case LayerConvolutional:
			layerNonActivated, err = gorgonia.Conv2d(lastActivatedLayer, net.Layers[i].WeightNode, tensor.Shape{net.Layers[i].KernelHeight, net.Layers[i].KernelWidth}, net.Layers[i].Padding, net.Layers[i].Stride, net.Layers[i].Dilation)
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't convolve[2D] input by kernel of Generator's layer #%d", i))
			}
			break
		case LayerMaxpool:
			layerNonActivated, err = gorgonia.MaxPool2D(lastActivatedLayer, tensor.Shape{net.Layers[i].KernelHeight, net.Layers[i].KernelWidth}, net.Layers[i].Padding, net.Layers[i].Stride)
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't maxpool[2D] input by kernel of Generator's layer #%d", i))
			}
			break
		case LayerFlatten:
			layerNonActivated, err = gorgonia.Reshape(lastActivatedLayer, tensor.Shape{batchSize, lastActivatedLayer.Shape().TotalSize() / batchSize})
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't flatten input of Generator's layer #%d", i))
			}
			break
		case LayerReshape:
			layerNonActivated, err = gorgonia.Reshape(lastActivatedLayer, net.Layers[i].ReshapeDims)
			if err != nil {
				return errors.Wrap(err, fmt.Sprintf("Can't reshape input of Generator's layer #%d", i))
			}
			break
		default:
			return fmt.Errorf("Layer #%d's type '%d' (uint16) is not handled [Generator]", i, net.Layers[i].Type)
		}

		gorgonia.WithName(fmt.Sprintf("generator_%d", i))(layerNonActivated)
		if net.Layers[i].BiasNode != nil {
			if batchSize < 2 {
				layerNonActivated, err = gorgonia.Add(layerNonActivated, net.Layers[i].BiasNode)
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("Can't add bias to non-activated output of Generator's layer #%d", i))
				}
			} else {
				layerNonActivated, err = gorgonia.BroadcastAdd(layerNonActivated, net.Layers[i].BiasNode, nil, []byte{0})
				if err != nil {
					return errors.Wrap(err, fmt.Sprintf("Can't add bias [in broadcast term with batch_size = %d] to non-activated output of Generator's layer #%d", batchSize, i))
				}
			}
		}
		layerActivated, err := net.Layers[i].Activation(layerNonActivated)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("Can't apply activation function to non-activated output of Generator's layer #%d", i))
		}
		gorgonia.WithName(fmt.Sprintf("generator_activated_%d", i))(layerActivated)
		lastActivatedLayer = layerActivated
		if i == len(net.Layers)-1 {
			net.out = layerActivated
		}

		fmt.Println(layerNonActivated.Shape())
	}
	return nil
}
