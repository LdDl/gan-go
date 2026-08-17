package gan_go

import (
	"math"
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func checkFloat(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Errorf("%s: got %v, want finite value near %v", name, got, want)
		return
	}
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %.15f, want %.15f", name, got, want)
	}
}

// evalLoss builds a graph with nodes a and b of shape 1x2, applies provided loss function and returns its scalar value
func evalLoss(t *testing.T, lossFn func(a, b *gorgonia.Node) (*gorgonia.Node, error), aVals, bVals []float64) float64 {
	t.Helper()
	g := gorgonia.NewGraph()
	at := tensor.New(tensor.WithShape(1, 2), tensor.WithBacking(aVals))
	bt := tensor.New(tensor.WithShape(1, 2), tensor.WithBacking(bVals))
	a := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 2), gorgonia.WithName("a"), gorgonia.WithValue(at))
	b := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 2), gorgonia.WithName("b"), gorgonia.WithValue(bt))
	loss, err := lossFn(a, b)
	if err != nil {
		t.Fatalf("loss build error: %v", err)
	}
	vm := gorgonia.NewTapeMachine(g)
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		t.Fatalf("vm error: %v", err)
	}
	return loss.Value().Data().(float64)
}

var (
	lossA = []float64{0.8, 0.2}
	lossB = []float64{1.0, 0.0}
)

func TestMSELoss(t *testing.T) {
	got := evalLoss(t, func(a, b *gorgonia.Node) (*gorgonia.Node, error) {
		return MSELoss(a, b)
	}, lossA, lossB)
	checkFloat(t, "MSE mean", got, 0.04, 1e-12)
}

func TestL1Loss(t *testing.T) {
	got := evalLoss(t, func(a, b *gorgonia.Node) (*gorgonia.Node, error) {
		return L1Loss(a, b)
	}, lossA, lossB)
	checkFloat(t, "L1 mean", got, 0.2, 1e-12)
}

func TestCrossEntropyLoss(t *testing.T) {
	got := evalLoss(t, func(a, b *gorgonia.Node) (*gorgonia.Node, error) {
		return CrossEntropyLoss(a, b)
	}, lossA, lossB)
	checkFloat(t, "CE mean", got, -math.Log(0.8)/2.0, 1e-9)
}

func TestBinaryCrossEntropyLoss(t *testing.T) {
	gotMean := evalLoss(t, func(a, b *gorgonia.Node) (*gorgonia.Node, error) {
		return BinaryCrossEntropyLoss(a, b)
	}, lossA, lossB)
	checkFloat(t, "BCE mean", gotMean, -math.Log(0.8), 1e-9)

	gotSum := evalLoss(t, func(a, b *gorgonia.Node) (*gorgonia.Node, error) {
		return BinaryCrossEntropyLoss(a, b, LossReductionSum)
	}, lossA, lossB)
	checkFloat(t, "BCE sum", gotSum, -2.0*math.Log(0.8), 1e-9)
}

func TestHuberLoss(t *testing.T) {
	delta := 2.0
	x := 0.2
	want := delta * delta * (math.Sqrt(1.0+(x/delta)*(x/delta)) - 1.0)
	got := evalLoss(t, func(a, b *gorgonia.Node) (*gorgonia.Node, error) {
		return HuberLoss(a, b, delta)
	}, lossA, lossB)
	checkFloat(t, "Huber mean float64", got, want, 1e-12)
}

func TestHuberLossFloat32(t *testing.T) {
	delta := 2.0
	x := 0.2
	want := delta * delta * (math.Sqrt(1.0+(x/delta)*(x/delta)) - 1.0)

	g := gorgonia.NewGraph()
	at := tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{0.8, 0.2}))
	bt := tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{1.0, 0.0}))
	a := gorgonia.NewMatrix(g, gorgonia.Float32, gorgonia.WithShape(1, 2), gorgonia.WithName("a32"), gorgonia.WithValue(at))
	b := gorgonia.NewMatrix(g, gorgonia.Float32, gorgonia.WithShape(1, 2), gorgonia.WithName("b32"), gorgonia.WithValue(bt))
	loss, err := HuberLoss(a, b, float32(2.0))
	if err != nil {
		t.Fatalf("loss build error: %v", err)
	}
	vm := gorgonia.NewTapeMachine(g)
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		t.Fatalf("vm error: %v", err)
	}
	checkFloat(t, "Huber mean float32", float64(loss.Value().Data().(float32)), want, 1e-6)
}

// TestHuberLossSameGraph guards against Gorgonia's deduplication of input nodes:
// two loss nodes with different deltas must coexist on the same graph
func TestHuberLossSameGraph(t *testing.T) {
	g := gorgonia.NewGraph()
	mk := func(name string, vals []float64) *gorgonia.Node {
		tt := tensor.New(tensor.WithShape(1, 2), tensor.WithBacking(vals))
		return gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 2), gorgonia.WithName(name), gorgonia.WithValue(tt))
	}
	a1, b1 := mk("ha1", lossA), mk("hb1", lossB)
	a2, b2 := mk("ha2", lossA), mk("hb2", lossB)
	loss1, err := HuberLoss(a1, b1, 2.0)
	if err != nil {
		t.Fatal(err)
	}
	loss2, err := HuberLoss(a2, b2, 3.0)
	if err != nil {
		t.Fatal(err)
	}
	vm := gorgonia.NewTapeMachine(g)
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		t.Fatal(err)
	}
	want1 := 4.0 * (math.Sqrt(1.0+0.01) - 1.0)
	want2 := 9.0 * (math.Sqrt(1.0+(0.2/3.0)*(0.2/3.0)) - 1.0)
	checkFloat(t, "Huber same graph delta=2", loss1.Value().Data().(float64), want1, 1e-12)
	checkFloat(t, "Huber same graph delta=3", loss2.Value().Data().(float64), want2, 1e-12)
}

// TestBCESaturated checks that exactly saturated activations produce finite loss values
func TestBCESaturated(t *testing.T) {
	perfect := evalLoss(t, func(a, b *gorgonia.Node) (*gorgonia.Node, error) {
		return BinaryCrossEntropyLoss(a, b)
	}, []float64{1.0, 0.0}, []float64{1.0, 0.0})
	checkFloat(t, "BCE saturated perfect", perfect, 0.0, 1e-9)

	wrong := evalLoss(t, func(a, b *gorgonia.Node) (*gorgonia.Node, error) {
		return BinaryCrossEntropyLoss(a, b)
	}, []float64{0.0, 1.0}, []float64{1.0, 0.0})
	checkFloat(t, "BCE saturated wrong", wrong, -math.Log(1e-12), 1e-6)
}

// TestCESaturatedGradients checks that gradients through saturated cross entropy stay finite
func TestCESaturatedGradients(t *testing.T) {
	g := gorgonia.NewGraph()
	at := tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{1.0, 0.0}))
	bt := tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{0.0, 1.0}))
	a := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 2), gorgonia.WithName("sat_a"), gorgonia.WithValue(at))
	b := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 2), gorgonia.WithName("sat_b"), gorgonia.WithValue(bt))
	loss, err := CrossEntropyLoss(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gorgonia.Grad(loss, a); err != nil {
		t.Fatal(err)
	}
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(a))
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		t.Fatal(err)
	}
	checkFloat(t, "CE saturated wrong", loss.Value().Data().(float64), -math.Log(1e-12)/2.0, 1e-6)
	gradNode, err := a.Grad()
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range gradNode.Data().([]float64) {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("gradient element %d is not finite: %v", i, v)
		}
	}
}
