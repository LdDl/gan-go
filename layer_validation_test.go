package gan_go

import (
	"testing"

	"gorgonia.org/gorgonia"
)

func TestDropoutProbabilityValidation(t *testing.T) {
	g := gorgonia.NewGraph()
	in := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 2), gorgonia.WithName("dropin"), gorgonia.WithInit(gorgonia.Ones()))
	for _, probability := range []float64{1.5, -0.1} {
		layer := &DropoutLayer{
			Probability: probability,
		}
		if _, err := layer.Fwd(1, in); err == nil {
			t.Errorf("dropout probability %v accepted, expected error", probability)
		}
	}
}

func TestLinearNilWeights(t *testing.T) {
	g := gorgonia.NewGraph()
	in := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 2), gorgonia.WithName("linin"), gorgonia.WithInit(gorgonia.Ones()))
	layer := &LinearLayer{
		Activation: NoActivation,
	}
	if _, err := layer.Fwd(1, in); err == nil {
		t.Error("linear layer with nil weights accepted, expected error")
	}
}

func TestSingleInputValidation(t *testing.T) {
	g := gorgonia.NewGraph()
	in := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 2), gorgonia.WithName("flin"), gorgonia.WithInit(gorgonia.Ones()))
	layer := &FlattenLayer{}
	if _, err := layer.Fwd(1); err == nil {
		t.Error("no inputs accepted, expected error")
	}
	if _, err := layer.Fwd(1, in, in); err == nil {
		t.Error("two inputs accepted, expected error")
	}
}
