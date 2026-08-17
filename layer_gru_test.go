package gan_go

import (
	"math"
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// refGRU computes GRU forward pass with plain loops: x [seq][feat], W [feat][3H], U [H][3H], b [3H].
// Gate columns order matches GRULayer: reset, update, candidate.
// Bias is applied to the input projection only. Reset gate is applied to the previous hidden state
// before the projection (formulation of Cho et al., 2014)
func refGRU(x, W, U [][]float64, b []float64, H int) [][]float64 {
	feat := len(x[0])
	h := make([]float64, H)
	out := make([][]float64, len(x))
	for t := range x {
		xProj := make([]float64, 3*H)
		hProj := make([]float64, 3*H)
		for j := 0; j < 3*H; j++ {
			s := b[j]
			for k := 0; k < feat; k++ {
				s += x[t][k] * W[k][j]
			}
			xProj[j] = s
			s = 0.0
			for k := 0; k < H; k++ {
				s += h[k] * U[k][j]
			}
			hProj[j] = s
		}
		resetGate := make([]float64, H)
		updateGate := make([]float64, H)
		for k := 0; k < H; k++ {
			resetGate[k] = sigmoidRef(xProj[k] + hProj[k])
			updateGate[k] = sigmoidRef(xProj[H+k] + hProj[H+k])
		}
		newH := make([]float64, H)
		for k := 0; k < H; k++ {
			gatedProj := 0.0
			for j := 0; j < H; j++ {
				gatedProj += resetGate[j] * h[j] * U[j][2*H+k]
			}
			candidate := math.Tanh(xProj[2*H+k] + gatedProj)
			newH[k] = (1.0-updateGate[k])*candidate + updateGate[k]*h[k]
		}
		h = newH
		out[t] = newH
	}
	return out
}

func buildGRUFixture(t *testing.T) (*gorgonia.ExprGraph, *GRULayer, *gorgonia.Node, [][]float64) {
	t.Helper()
	const (
		seq  = 3
		feat = 2
		H    = 2
	)
	wData := make([]float64, feat*3*H)
	uData := make([]float64, H*3*H)
	bData := make([]float64, 3*H)
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
		w2d[k] = wData[k*3*H : (k+1)*3*H]
	}
	u2d := make([][]float64, H)
	for k := 0; k < H; k++ {
		u2d[k] = uData[k*3*H : (k+1)*3*H]
	}
	want := refGRU(x2d, w2d, u2d, bData, H)

	g := gorgonia.NewGraph()
	wNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(feat, 3*H), gorgonia.WithName("gru_w"), gorgonia.WithValue(tensor.New(tensor.WithShape(feat, 3*H), tensor.WithBacking(wData))))
	uNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(H, 3*H), gorgonia.WithName("gru_u"), gorgonia.WithValue(tensor.New(tensor.WithShape(H, 3*H), tensor.WithBacking(uData))))
	bNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 3*H), gorgonia.WithName("gru_b"), gorgonia.WithValue(tensor.New(tensor.WithShape(1, 3*H), tensor.WithBacking(bData))))
	xNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(seq, feat), gorgonia.WithName("gru_x"), gorgonia.WithValue(tensor.New(tensor.WithShape(seq, feat), tensor.WithBacking(xData))))

	layer := &GRULayer{
		InputWeightNode:  wNode,
		HiddenWeightNode: uNode,
		BiasNode:         bNode,
		HiddenSize:       H,
	}
	return g, layer, xNode, want
}

func TestGRUForward(t *testing.T) {
	g, layer, xNode, want := buildGRUFixture(t)
	out, err := layer.Fwd(1, xNode)
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

func TestGRUSolverStep(t *testing.T) {
	g, layer, xNode, _ := buildGRUFixture(t)
	out, err := layer.Fwd(1, xNode)
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

// TestGRUHiddenSizeOne guards against dimension collapse when slicing ranges of width 1
func TestGRUHiddenSizeOne(t *testing.T) {
	const (
		seq  = 2
		feat = 2
		H    = 1
	)
	wData := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	uData := []float64{0.7, 0.8, 0.9}
	bData := []float64{0.01, 0.02, 0.03}
	xData := []float64{0.5, 0.6, 0.7, 0.8}

	want := refGRU(
		[][]float64{xData[0:2], xData[2:4]},
		[][]float64{wData[0:3], wData[3:6]},
		[][]float64{uData},
		bData,
		H,
	)

	g := gorgonia.NewGraph()
	wNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(feat, 3*H), gorgonia.WithName("gru1_w"), gorgonia.WithValue(tensor.New(tensor.WithShape(feat, 3*H), tensor.WithBacking(wData))))
	uNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(H, 3*H), gorgonia.WithName("gru1_u"), gorgonia.WithValue(tensor.New(tensor.WithShape(H, 3*H), tensor.WithBacking(uData))))
	bNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 3*H), gorgonia.WithName("gru1_b"), gorgonia.WithValue(tensor.New(tensor.WithShape(1, 3*H), tensor.WithBacking(bData))))
	xNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(seq, feat), gorgonia.WithName("gru1_x"), gorgonia.WithValue(tensor.New(tensor.WithShape(seq, feat), tensor.WithBacking(xData))))

	layer := &GRULayer{
		InputWeightNode:  wNode,
		HiddenWeightNode: uNode,
		BiasNode:         bNode,
		HiddenSize:       H,
	}
	out, err := layer.Fwd(1, xNode)
	if err != nil {
		t.Fatalf("Fwd error: %v", err)
	}
	vm := gorgonia.NewTapeMachine(g)
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		t.Fatalf("vm error: %v", err)
	}
	got := out.Value().Data().([]float64)
	for i := 0; i < seq; i++ {
		checkFloat(t, "hidden state", got[i], want[i][0], 1e-12)
	}
}

func TestGRUCloneTo(t *testing.T) {
	_, layer, _, _ := buildGRUFixture(t)
	g2 := gorgonia.NewGraph()
	clonedIface, err := layer.CloneTo(g2, "_clone")
	if err != nil {
		t.Fatalf("CloneTo error: %v", err)
	}
	cloned := clonedIface.(*GRULayer)
	if got := len(cloned.Learnables()); got != 3 {
		t.Errorf("clone learnables count: got %d, want 3", got)
	}
	layer.InputWeightNode.Value().Data().([]float64)[0] = 42.0
	if got := cloned.InputWeightNode.Value().Data().([]float64)[0]; got != 42.0 {
		t.Errorf("clone does not share weight memory: got %v, want 42.0", got)
	}
}
