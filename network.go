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
type Network struct {
	Name   string
	Layers []Layer
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
			learnables = append(learnables, l.Learnables()...)
		}
	}
	return learnables
}

// Fwd Initializates feedforward for provided input
//
// inputs - Input node (or nodes). Only first layer of network can accept multiple inputs
// batchSize - batch size. If it's >= 2 then broadcast function will be applied
func (net *Network) Fwd(batchSize int, inputs ...*gorgonia.Node) error {
	if len(inputs) == 0 {
		return fmt.Errorf("There are no input nodes for network")
	}

	networkName := "network"
	if net.Name != "" {
		networkName = net.Name
	}

	if len(net.Layers) == 0 {
		return fmt.Errorf("Network must have one layer atleast")
	}

	// Feedforward input through the layers
	layerInputs := inputs
	for i, layer := range net.Layers {
		if layer == nil {
			return fmt.Errorf("Network's layer #%d is nil", i)
		}
		layerNonActivated, err := layer.Fwd(batchSize, layerInputs...)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("[Network, Layer #%d] Can't feedforward input before activation", i))
		}
		gorgonia.WithName(fmt.Sprintf("%s_%d", networkName, i))(layerNonActivated)
		layerActivated, err := layer.Activate(layerNonActivated)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("Can't apply activation function to non-activated output of Network's layer #%d", i))
		}
		gorgonia.WithName(fmt.Sprintf("%s_activated_%d", networkName, i))(layerActivated)
		layerInputs = gorgonia.Nodes{layerActivated}
	}
	net.out = layerInputs[0]
	return nil
}
