package gan_go

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

// Network Abstraction for neural network.
//
// Layers - simple sequence of layers
// out - alias to activated output of last layer
//
type Network struct {
	Name   string
	Layers []*Layer
	out    *gorgonia.Node
}

// Out Returns reference to output node
func (net *Network) Out() *gorgonia.Node {
	return net.out
}

// Learnables Returns learnables nodes
func (net *Network) Learnables() gorgonia.Nodes {
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
func (net *Network) Fwd(input *gorgonia.Node, batchSize int) error {
	var err error

	networkName := "network"
	if net.Name != "" {
		networkName = net.Name
	}

	if len(net.Layers) == 0 {
		return fmt.Errorf("Network must have one layer atleast")
	}
	if net.Layers[0] == nil {
		return fmt.Errorf("Network's layer #0 is nil")
	}

	// Feedforward input through first layer
	firstLayerNonActivated, err := net.Layers[0].Fwd(batchSize, input)
	if err != nil {
		return errors.Wrap(err, "[Network, Layer #0] Can't feedforward input before activation")
	}
	gorgonia.WithName(fmt.Sprintf("%s_0", networkName))(firstLayerNonActivated)
	// Activate first layer's output
	firstLayerActivated, err := net.Layers[0].Activation(firstLayerNonActivated)
	if err != nil {
		return errors.Wrap(err, "Can't apply activation function to non-activated output of Network's layer #0")
	}
	gorgonia.WithName(fmt.Sprintf("%s_activated_0", networkName))(firstLayerActivated)
	lastActivatedLayer := firstLayerActivated

	if len(net.Layers) == 1 {
		net.out = lastActivatedLayer
	}

	// Feedforward input through remaining layers
	for i := 1; i < len(net.Layers); i++ {
		if net.Layers[i] == nil {
			return fmt.Errorf("Network's layer #%d is nil", i)
		}
		if net.Layers[i].WeightNode == nil && !noWeightsAllowed(net.Layers[i].Type) {
			return fmt.Errorf("Network's layer's #%d WeightNode is nil", i)
		}
		// Feedforward input through i-th layer (i != 0)
		layerNonActivated, err := net.Layers[i].Fwd(batchSize, lastActivatedLayer)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("[Network, Layer #%d] Can't feedforward input before activation", i))
		}
		gorgonia.WithName(fmt.Sprintf("%s_%d", networkName, i))(layerNonActivated)
		// Activate i-th layer's output (i != 0)
		layerActivated, err := net.Layers[i].Activation(layerNonActivated)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("Can't apply activation function to non-activated output of Network's layer #%d", i))
		}
		gorgonia.WithName(fmt.Sprintf("%s_activated_%d", networkName, i))(layerActivated)
		lastActivatedLayer = layerActivated
		if i == len(net.Layers)-1 {
			net.out = layerActivated
		}
	}
	return nil
}
