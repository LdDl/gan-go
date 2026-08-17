package gan_go

import (
	"math"
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// refRNN computes vanilla RNN forward pass with plain loops: x [seq][feat], W [feat][H], U [H][H], b [H]
func refRNN(x, W, U [][]float64, b []float64, H int) [][]float64 {
	feat := len(x[0])
	h := make([]float64, H)
	out := make([][]float64, len(x))
	for t := range x {
		newH := make([]float64, H)
		for j := 0; j < H; j++ {
			s := b[j]
			for k := 0; k < feat; k++ {
				s += x[t][k] * W[k][j]
			}
			for k := 0; k < H; k++ {
				s += h[k] * U[k][j]
			}
			newH[j] = math.Tanh(s)
		}
		h = newH
		out[t] = newH
	}
	return out
}

func buildRNNFixture(t *testing.T) (*gorgonia.ExprGraph, *RNNLayer, *gorgonia.Node, [][]float64) {
	t.Helper()
	const (
		seq  = 3
		feat = 2
		H    = 2
	)
	wData := make([]float64, feat*H)
	uData := make([]float64, H*H)
	bData := make([]float64, H)
	xData := make([]float64, seq*feat)
	for i := range wData {
		wData[i] = math.Sin(float64(i)+1.0) / 2.0
	}
	for i := range uData {
		uData[i] = math.Cos(float64(i)+1.0) / 2.0
	}
	for i := range bData {
		bData[i] = math.Sin(float64(i)*2.0) / 4.0
	}
	for i := range xData {
		xData[i] = math.Cos(float64(i)*3.0) / 2.0
	}

	x2d := make([][]float64, seq)
	for i := 0; i < seq; i++ {
		x2d[i] = xData[i*feat : (i+1)*feat]
	}
	w2d := make([][]float64, feat)
	for k := 0; k < feat; k++ {
		w2d[k] = wData[k*H : (k+1)*H]
	}
	u2d := make([][]float64, H)
	for k := 0; k < H; k++ {
		u2d[k] = uData[k*H : (k+1)*H]
	}
	want := refRNN(x2d, w2d, u2d, bData, H)

	g := gorgonia.NewGraph()
	wNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(feat, H), gorgonia.WithName("rnn_w"), gorgonia.WithValue(tensor.New(tensor.WithShape(feat, H), tensor.WithBacking(wData))))
	uNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(H, H), gorgonia.WithName("rnn_u"), gorgonia.WithValue(tensor.New(tensor.WithShape(H, H), tensor.WithBacking(uData))))
	bNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, H), gorgonia.WithName("rnn_b"), gorgonia.WithValue(tensor.New(tensor.WithShape(1, H), tensor.WithBacking(bData))))
	xNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(seq, feat), gorgonia.WithName("rnn_x"), gorgonia.WithValue(tensor.New(tensor.WithShape(seq, feat), tensor.WithBacking(xData))))

	layer := &RNNLayer{
		InputWeightNode:  wNode,
		HiddenWeightNode: uNode,
		BiasNode:         bNode,
		HiddenSize:       H,
	}
	return g, layer, xNode, want
}

func TestRNNForward(t *testing.T) {
	g, layer, xNode, want := buildRNNFixture(t)
	out, err := layer.Fwd(xNode)
	if err != nil {
		t.Fatalf("Fwd error: %v", err)
	}
	vm := gorgonia.NewTapeMachine(g)
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		t.Fatalf("vm error: %v", err)
	}
	seq := len(want)
	H := layer.HiddenSize
	outShape := out.Value().(tensor.Tensor).Shape()
	if len(outShape) != 2 || outShape[0] != seq || outShape[1] != H {
		t.Fatalf("output shape: got %v, want [%d %d]", outShape, seq, H)
	}
	got := out.Value().Data().([]float64)
	for i := 0; i < seq; i++ {
		for k := 0; k < H; k++ {
			checkFloat(t, "hidden state", got[i*H+k], want[i][k], 1e-12)
		}
	}
	finalHidden := layer.FinalHidden().Value().Data().([]float64)
	for k := 0; k < H; k++ {
		checkFloat(t, "final hidden state", finalHidden[k], want[seq-1][k], 1e-12)
	}
}

func TestRNNSolverStep(t *testing.T) {
	g, layer, xNode, _ := buildRNNFixture(t)
	out, err := layer.Fwd(xNode)
	if err != nil {
		t.Fatalf("Fwd error: %v", err)
	}
	cost, err := gorgonia.Mean(out)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gorgonia.Grad(cost, layer.Learnables()...); err != nil {
		t.Fatalf("Grad error: %v", err)
	}
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(layer.Learnables()...))
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		t.Fatalf("vm error: %v", err)
	}
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(0.1))
	if err := solver.Step(gorgonia.NodesToValueGrads(layer.Learnables())); err != nil {
		t.Fatalf("solver step error: %v", err)
	}
}

func TestRNNCloneTo(t *testing.T) {
	_, layer, _, _ := buildRNNFixture(t)
	g2 := gorgonia.NewGraph()
	clonedIface, err := layer.CloneTo(g2, "_clone")
	if err != nil {
		t.Fatalf("CloneTo error: %v", err)
	}
	cloned := clonedIface.(*RNNLayer)
	if got := len(cloned.Learnables()); got != 3 {
		t.Errorf("clone learnables count: got %d, want 3", got)
	}
	layer.InputWeightNode.Value().Data().([]float64)[0] = 42.0
	if got := cloned.InputWeightNode.Value().Data().([]float64)[0]; got != 42.0 {
		t.Errorf("clone does not share weight memory: got %v, want 42.0", got)
	}
}
