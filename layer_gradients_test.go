package gan_go

import (
	"math"
	"testing"

	"gorgonia.org/gorgonia"
)

// numericGradCheck compares analytic gradients of the loss w.r.t. learnables
// against central finite differences. It guards against silently broken backward passes
// (e.g. wrong gradients of slice/reshape/activation combinations inside recurrent layers)
func numericGradCheck(t *testing.T, g *gorgonia.ExprGraph, out *gorgonia.Node, learnables gorgonia.Nodes) {
	t.Helper()
	cost, err := gorgonia.Mean(out)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gorgonia.Grad(cost, learnables...); err != nil {
		t.Fatalf("Grad error: %v", err)
	}
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(learnables...))
	defer vm.Close()

	run := func() float64 {
		if err := vm.RunAll(); err != nil {
			t.Fatalf("vm error: %v", err)
		}
		c := costVal.Data().(float64)
		vm.Reset()
		return c
	}
	run()

	// Snapshot of analytic gradients must be taken for ALL learnables right after the very first run:
	// tape machine accumulates gradients over RunAll calls, so values read after
	// the finite difference runs below would be summed multiple times
	analytic := make([][]float64, len(learnables))
	for i, learnable := range learnables {
		grad, err := learnable.Grad()
		if err != nil {
			t.Fatalf("no gradient for %s: %v", learnable.Name(), err)
		}
		analytic[i] = append([]float64{}, grad.Data().([]float64)...)
	}

	const eps = 1e-6
	for li, learnable := range learnables {
		backing := learnable.Value().Data().([]float64)
		for i := range backing {
			orig := backing[i]
			backing[i] = orig + eps
			costPlus := run()
			backing[i] = orig - eps
			costMinus := run()
			backing[i] = orig
			numeric := (costPlus - costMinus) / (2 * eps)
			if math.Abs(numeric-analytic[li][i]) > 1e-6 {
				t.Errorf("%s gradient [%d]: analytic %v, numeric %v", learnable.Name(), i, analytic[li][i], numeric)
			}
		}
	}
}

func TestLSTMGradients(t *testing.T) {
	g, layer, xNode, _ := buildLSTMFixture(t)
	out, err := layer.Fwd(1, xNode)
	if err != nil {
		t.Fatalf("Fwd error: %v", err)
	}
	numericGradCheck(t, g, out, layer.Learnables())
}

func TestGRUGradients(t *testing.T) {
	g, layer, xNode, _ := buildGRUFixture(t)
	out, err := layer.Fwd(1, xNode)
	if err != nil {
		t.Fatalf("Fwd error: %v", err)
	}
	numericGradCheck(t, g, out, layer.Learnables())
}

func TestRNNGradients(t *testing.T) {
	g, layer, xNode, _ := buildRNNFixture(t)
	out, err := layer.Fwd(1, xNode)
	if err != nil {
		t.Fatalf("Fwd error: %v", err)
	}
	numericGradCheck(t, g, out, layer.Learnables())
}
