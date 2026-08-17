package gan_go

import (
	"math"
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func sigmoidRef(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// refLSTM computes LSTM forward pass with plain loops: x [seq][feat], W [feat][4H], U [H][4H], b [4H].
// Gate columns order matches LSTMLayer: input, forget, cell candidate, output
func refLSTM(x, W, U [][]float64, b []float64, H int) [][]float64 {
	feat := len(x[0])
	h := make([]float64, H)
	c := make([]float64, H)
	out := make([][]float64, len(x))
	for t := range x {
		gates := make([]float64, 4*H)
		for j := 0; j < 4*H; j++ {
			s := b[j]
			for k := 0; k < feat; k++ {
				s += x[t][k] * W[k][j]
			}
			for k := 0; k < H; k++ {
				s += h[k] * U[k][j]
			}
			gates[j] = s
		}
		newH := make([]float64, H)
		newC := make([]float64, H)
		for k := 0; k < H; k++ {
			inputGate := sigmoidRef(gates[k])
			forgetGate := sigmoidRef(gates[H+k])
			cellCandidate := math.Tanh(gates[2*H+k])
			outputGate := sigmoidRef(gates[3*H+k])
			newC[k] = forgetGate*c[k] + inputGate*cellCandidate
			newH[k] = outputGate * math.Tanh(newC[k])
		}
		h, c = newH, newC
		out[t] = newH
	}
	return out
}

func buildLSTMFixture(t *testing.T) (*gorgonia.ExprGraph, *LSTMLayer, *gorgonia.Node, [][]float64) {
	t.Helper()
	const (
		seq  = 3
		feat = 2
		H    = 2
	)
	wData := make([]float64, feat*4*H)
	uData := make([]float64, H*4*H)
	bData := make([]float64, 4*H)
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
		w2d[k] = wData[k*4*H : (k+1)*4*H]
	}
	u2d := make([][]float64, H)
	for k := 0; k < H; k++ {
		u2d[k] = uData[k*4*H : (k+1)*4*H]
	}
	want := refLSTM(x2d, w2d, u2d, bData, H)

	g := gorgonia.NewGraph()
	wNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(feat, 4*H), gorgonia.WithName("lstm_w"), gorgonia.WithValue(tensor.New(tensor.WithShape(feat, 4*H), tensor.WithBacking(wData))))
	uNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(H, 4*H), gorgonia.WithName("lstm_u"), gorgonia.WithValue(tensor.New(tensor.WithShape(H, 4*H), tensor.WithBacking(uData))))
	bNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 4*H), gorgonia.WithName("lstm_b"), gorgonia.WithValue(tensor.New(tensor.WithShape(1, 4*H), tensor.WithBacking(bData))))
	xNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(seq, feat), gorgonia.WithName("lstm_x"), gorgonia.WithValue(tensor.New(tensor.WithShape(seq, feat), tensor.WithBacking(xData))))

	layer := &LSTMLayer{
		InputWeightNode:  wNode,
		HiddenWeightNode: uNode,
		BiasNode:         bNode,
		HiddenSize:       H,
	}
	return g, layer, xNode, want
}

func TestLSTMForward(t *testing.T) {
	g, layer, xNode, want := buildLSTMFixture(t)
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

func TestLSTMSolverStep(t *testing.T) {
	g, layer, xNode, _ := buildLSTMFixture(t)
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

// TestLSTMHiddenSizeOne guards against dimension collapse when slicing ranges of width 1
func TestLSTMHiddenSizeOne(t *testing.T) {
	const (
		seq  = 2
		feat = 2
		H    = 1
	)
	wData := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	uData := []float64{0.15, 0.25, 0.35, 0.45}
	bData := []float64{0.01, 0.02, 0.03, 0.04}
	xData := []float64{0.5, 0.6, 0.7, 0.8}

	want := refLSTM(
		[][]float64{xData[0:2], xData[2:4]},
		[][]float64{wData[0:4], wData[4:8]},
		[][]float64{uData},
		bData,
		H,
	)

	g := gorgonia.NewGraph()
	wNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(feat, 4*H), gorgonia.WithName("lstm1_w"), gorgonia.WithValue(tensor.New(tensor.WithShape(feat, 4*H), tensor.WithBacking(wData))))
	uNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(H, 4*H), gorgonia.WithName("lstm1_u"), gorgonia.WithValue(tensor.New(tensor.WithShape(H, 4*H), tensor.WithBacking(uData))))
	bNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, 4*H), gorgonia.WithName("lstm1_b"), gorgonia.WithValue(tensor.New(tensor.WithShape(1, 4*H), tensor.WithBacking(bData))))
	xNode := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(seq, feat), gorgonia.WithName("lstm1_x"), gorgonia.WithValue(tensor.New(tensor.WithShape(seq, feat), tensor.WithBacking(xData))))

	layer := &LSTMLayer{
		InputWeightNode:  wNode,
		HiddenWeightNode: uNode,
		BiasNode:         bNode,
		HiddenSize:       H,
	}
	out, err := layer.Fwd(xNode)
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

func TestLSTMCloneTo(t *testing.T) {
	_, layer, _, _ := buildLSTMFixture(t)
	g2 := gorgonia.NewGraph()
	clonedIface, err := layer.CloneTo(g2, "_clone")
	if err != nil {
		t.Fatalf("CloneTo error: %v", err)
	}
	cloned := clonedIface.(*LSTMLayer)
	if got := len(cloned.Learnables()); got != 3 {
		t.Errorf("clone learnables count: got %d, want 3", got)
	}
	layer.InputWeightNode.Value().Data().([]float64)[0] = 42.0
	if got := cloned.InputWeightNode.Value().Data().([]float64)[0]; got != 42.0 {
		t.Errorf("clone does not share weight memory: got %v, want 42.0", got)
	}
}
