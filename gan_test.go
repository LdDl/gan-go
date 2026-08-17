package gan_go

import (
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func mk11(g *gorgonia.ExprGraph, name string, value float64) *gorgonia.Node {
	backing := tensor.New(tensor.WithShape(1, 1), tensor.WithBacking([]float64{value}))
	return gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 1), gorgonia.WithName(name), gorgonia.WithValue(backing))
}

// TestSharedTensorAssumption documents and guards the core trick of NewGAN:
// gorgonia.WithValue binds the same underlying tensor to the new node,
// and solvers update weights in place, so the copy sees every update of the source
func TestSharedTensorAssumption(t *testing.T) {
	g1 := gorgonia.NewGraph()
	w1 := mk11(g1, "w1", 2.0)

	g2 := gorgonia.NewGraph()
	w2 := gorgonia.NewMatrix(g2, gorgonia.Float64, gorgonia.WithShape(1, 1), gorgonia.WithName("w2"), gorgonia.WithValue(w1.Value()))

	// Direct in place mutation of the source must be visible in the copy
	w1.Value().Data().([]float64)[0] = 10.0
	if got := w2.Value().Data().([]float64)[0]; got != 10.0 {
		t.Fatalf("copy does not share backing memory: got %v, want 10.0", got)
	}

	// Solver step on the source graph must be visible in the copy as well.
	// cost = w1^2, gradient is 2*w1 = 20, vanilla SGD with lr 0.1 gives w1 = 10 - 0.1*20 = 8
	sq, err := gorgonia.Square(w1)
	if err != nil {
		t.Fatal(err)
	}
	cost, err := gorgonia.Sum(sq)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gorgonia.Grad(cost, w1); err != nil {
		t.Fatal(err)
	}
	vm := gorgonia.NewTapeMachine(g1, gorgonia.BindDualValues(w1))
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		t.Fatal(err)
	}
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(0.1))
	if err := solver.Step(gorgonia.NodesToValueGrads(gorgonia.Nodes{w1})); err != nil {
		t.Fatal(err)
	}
	if got := w1.Value().Data().([]float64)[0]; got != 8.0 {
		t.Fatalf("solver did not update source in place: got %v, want 8.0", got)
	}
	if got := w2.Value().Data().([]float64)[0]; got != 8.0 {
		t.Fatalf("copy does not see solver update: got %v, want 8.0", got)
	}
}

func TestNewGANSharedWeights(t *testing.T) {
	ganGraph := gorgonia.NewGraph()
	disGraph := gorgonia.NewGraph()

	genW := mk11(ganGraph, "gen_w", 1.0)
	generator := Generator(
		&LinearLayer{
			WeightNode: genW,
			Activation: NoActivation,
		},
	)
	genInput := mk11(ganGraph, "gen_in", 1.0)
	if err := generator.Fwd(genInput); err != nil {
		t.Fatal(err)
	}

	disW := mk11(disGraph, "dis_w", 2.0)
	discriminator := Discriminator(
		&LinearLayer{
			WeightNode: disW,
			Activation: NoActivation,
		},
	)

	theGAN, err := NewGAN(ganGraph, generator, discriminator)
	if err != nil {
		t.Fatal(err)
	}
	if err := theGAN.Fwd(); err != nil {
		t.Fatal(err)
	}

	if got := len(theGAN.Learnables()); got != 2 {
		t.Errorf("GAN learnables count: got %d, want 2", got)
	}
	if got := len(theGAN.GeneratorLearnables()); got != 1 {
		t.Errorf("GAN generator learnables count: got %d, want 1", got)
	}

	// Mutate original discriminator weight in place, cloned node must see it
	disW.Value().Data().([]float64)[0] = 5.0
	clonedW := theGAN.Learnables()[1]
	if got := clonedW.Value().Data().([]float64)[0]; got != 5.0 {
		t.Errorf("cloned weight does not share memory: got %v, want 5.0", got)
	}

	// GAN forward must use the updated weight: out = in * genW * disW = 1 * 1 * 5
	vm := gorgonia.NewTapeMachine(ganGraph)
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		t.Fatal(err)
	}
	if got := theGAN.Out().Value().Data().([]float64)[0]; got != 5.0 {
		t.Errorf("GAN forward through cloned weights: got %v, want 5.0", got)
	}
}
