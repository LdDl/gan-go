package main

import (
	_ "embed"
	"fmt"
	"strings"
	"text/template"

	gan "github.com/LdDl/gan-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

//go:embed templates/linear.md.tmpl
var linearTemplate string

// linearFixture Payload of the linear layer example, both for markdown and animations
type linearFixture struct {
	Layer      string      `json:"layer"`
	Input      [][]float64 `json:"input"`
	Weights    [][]float64 `json:"weights"`
	Bias       []float64   `json:"bias"`
	Output     [][]float64 `json:"output"`
	GradOutput [][]float64 `json:"grad_output"`
	GradW      [][]float64 `json:"grad_weights"`
	GradB      []float64   `json:"grad_bias"`
	GradX      [][]float64 `json:"grad_input"`
	// Vanilla gradient descent step with LearningRate applied to weights and bias
	LearningRate float64     `json:"learning_rate"`
	NewWeights   [][]float64 `json:"new_weights"`
	NewBias      []float64   `json:"new_bias"`
}

func generateLinear() error {
	x := [][]float64{
		{1, 2, -1},
		{0, 3, 2},
	}
	w := [][]float64{
		{2, -1, 0},
		{1, 0, 3},
	}
	b := []float64{1, -2}
	delta := [][]float64{
		{1, -2},
		{0, 3},
	}
	batch := len(x)
	fIn := len(x[0])
	fOut := len(w)

	// Manual forward: y[i][j] = sum_k x[i][k]*w[j][k] + b[j]
	y := make([][]float64, batch)
	for i := 0; i < batch; i++ {
		y[i] = make([]float64, fOut)
		for j := 0; j < fOut; j++ {
			s := b[j]
			for k := 0; k < fIn; k++ {
				s += x[i][k] * w[j][k]
			}
			y[i][j] = s
		}
	}

	// Manual backward
	gradW := make([][]float64, fOut)
	for j := 0; j < fOut; j++ {
		gradW[j] = make([]float64, fIn)
		for k := 0; k < fIn; k++ {
			for i := 0; i < batch; i++ {
				gradW[j][k] += delta[i][j] * x[i][k]
			}
		}
	}
	gradB := make([]float64, fOut)
	for j := 0; j < fOut; j++ {
		for i := 0; i < batch; i++ {
			gradB[j] += delta[i][j]
		}
	}
	gradX := make([][]float64, batch)
	for i := 0; i < batch; i++ {
		gradX[i] = make([]float64, fIn)
		for k := 0; k < fIn; k++ {
			for j := 0; j < fOut; j++ {
				gradX[i][k] += delta[i][j] * w[j][k]
			}
		}
	}

	// Manual gradient descent step
	learningRate := 0.1
	newW := make([][]float64, fOut)
	for j := 0; j < fOut; j++ {
		newW[j] = make([]float64, fIn)
		for k := 0; k < fIn; k++ {
			newW[j][k] = w[j][k] - learningRate*gradW[j][k]
		}
	}
	newB := make([]float64, fOut)
	for j := 0; j < fOut; j++ {
		newB[j] = b[j] - learningRate*gradB[j]
	}

	if err := verifyLinear(x, w, b, delta, y, gradW, gradB, gradX, learningRate, newW, newB); err != nil {
		return err
	}

	fixture := linearFixture{
		Layer:        "linear",
		Input:        x,
		Weights:      w,
		Bias:         b,
		Output:       y,
		GradOutput:   delta,
		GradW:        gradW,
		GradB:        gradB,
		GradX:        gradX,
		LearningRate: learningRate,
		NewWeights:   newW,
		NewBias:      newB,
	}
	if err := writeJSON("linear", fixture); err != nil {
		return err
	}
	content, err := linearMarkdown(fixture)
	if err != nil {
		return err
	}
	return writeMarkdown("linear", content)
}

// verifyLinear replays the fixture through the actual layer, Gorgonia gradients and a solver step
func verifyLinear(x, w [][]float64, b []float64, delta, wantY, wantGW [][]float64, wantGB []float64, wantGX [][]float64, learningRate float64, wantNewW [][]float64, wantNewB []float64) error {
	g := gorgonia.NewGraph()
	xNode := matrixNode(g, "x", x)
	wNode := matrixNode(g, "w", w)
	bNode := matrixNode(g, "b", [][]float64{b})
	layer := &gan.LinearLayer{
		WeightNode: wNode,
		BiasNode:   bNode,
		Activation: gan.NoActivation,
	}
	out, err := layer.Fwd(xNode)
	if err != nil {
		return err
	}
	deltaNode := matrixNode(g, "delta", delta)
	weighted, err := gorgonia.HadamardProd(out, deltaNode)
	if err != nil {
		return err
	}
	cost, err := gorgonia.Sum(weighted)
	if err != nil {
		return err
	}
	learnables := gorgonia.Nodes{wNode, bNode, xNode}
	if _, err := gorgonia.Grad(cost, learnables...); err != nil {
		return err
	}
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(learnables...))
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		return err
	}
	if err := compareFlat("linear forward", out.Value().Data().([]float64), flatten2(wantY)); err != nil {
		return err
	}
	gw, err := wNode.Grad()
	if err != nil {
		return err
	}
	if err := compareFlat("linear dW", gw.Data().([]float64), flatten2(wantGW)); err != nil {
		return err
	}
	gb, err := bNode.Grad()
	if err != nil {
		return err
	}
	if err := compareFlat("linear db", gb.Data().([]float64), wantGB); err != nil {
		return err
	}
	gx, err := xNode.Grad()
	if err != nil {
		return err
	}
	if err := compareFlat("linear dx", gx.Data().([]float64), flatten2(wantGX)); err != nil {
		return err
	}
	// Solver step updates weights in place, must be the last check
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(learningRate))
	if err := solver.Step(gorgonia.NodesToValueGrads(gorgonia.Nodes{wNode, bNode})); err != nil {
		return err
	}
	if err := compareFlat("linear W after step", wNode.Value().Data().([]float64), flatten2(wantNewW)); err != nil {
		return err
	}
	return compareFlat("linear b after step", bNode.Value().Data().([]float64), wantNewB)
}

// expansionStep One fully expanded element for the template: left side index, expression body, result
type expansionStep struct {
	LHS    string
	Body   string
	Result string
}

// linearView View model for the linear template: all math is precomputed here,
// the template holds the prose and the document structure only
type linearView struct {
	Batch, FIn, FOut                                   int
	X, W, B, Y, Delta, GradW, GradB, GradX, NewW, NewB string
	Row0, Row1                                         string
	LR, Upd00, Upd12                                   string
	ForwardSteps                                       []expansionStep
	GradWSteps                                         []expansionStep
	GradBSteps                                         []expansionStep
	GradXSteps                                         []expansionStep
}

func linearMarkdown(f linearFixture) (string, error) {
	batch := len(f.Input)
	fIn := len(f.Input[0])
	fOut := len(f.Weights)

	view := linearView{
		Batch: batch,
		FIn:   fIn,
		FOut:  fOut,
		X:     texMatrix(f.Input),
		W:     texMatrix(f.Weights),
		B:     texMatrix([][]float64{f.Bias}),
		Y:     texMatrix(f.Output),
		Delta: texMatrix(f.GradOutput),
		GradW: texMatrix(f.GradW),
		GradB: texMatrix([][]float64{f.GradB}),
		GradX: texMatrix(f.GradX),
		NewW:  texMatrix(f.NewWeights),
		NewB:  texMatrix([][]float64{f.NewBias}),
		Row0:  rowList(f.Input[0]),
		Row1:  rowList(f.Input[1]),
		LR:    fmtNum(f.LearningRate),
		Upd00: fmt.Sprintf("%s - %s \\cdot %s = %s", wrapNeg(f.Weights[0][0]), fmtNum(f.LearningRate), wrapNeg(f.GradW[0][0]), fmtNum(f.NewWeights[0][0])),
		Upd12: fmt.Sprintf("%s - %s \\cdot %s = %s", wrapNeg(f.Weights[1][2]), fmtNum(f.LearningRate), wrapNeg(f.GradW[1][2]), fmtNum(f.NewWeights[1][2])),
	}
	for i := 0; i < batch; i++ {
		for j := 0; j < fOut; j++ {
			terms := make([]string, fIn)
			for k := 0; k < fIn; k++ {
				terms[k] = fmt.Sprintf("%s \\cdot %s", wrapNeg(f.Input[i][k]), wrapNeg(f.Weights[j][k]))
			}
			view.ForwardSteps = append(view.ForwardSteps, expansionStep{
				LHS:    fmt.Sprintf("%d%d", i, j),
				Body:   strings.Join(terms, " + ") + " + " + wrapNeg(f.Bias[j]),
				Result: fmtNum(f.Output[i][j]),
			})
		}
	}
	for j := 0; j < fOut; j++ {
		for k := 0; k < fIn; k++ {
			terms := make([]string, batch)
			for i := 0; i < batch; i++ {
				terms[i] = fmt.Sprintf("%s \\cdot %s", wrapNeg(f.GradOutput[i][j]), wrapNeg(f.Input[i][k]))
			}
			view.GradWSteps = append(view.GradWSteps, expansionStep{
				LHS:    fmt.Sprintf("%d%d", j, k),
				Body:   strings.Join(terms, " + "),
				Result: fmtNum(f.GradW[j][k]),
			})
		}
	}
	for j := 0; j < fOut; j++ {
		terms := make([]string, batch)
		for i := 0; i < batch; i++ {
			terms[i] = wrapNeg(f.GradOutput[i][j])
		}
		view.GradBSteps = append(view.GradBSteps, expansionStep{
			LHS:    fmt.Sprintf("%d", j),
			Body:   strings.Join(terms, " + "),
			Result: fmtNum(f.GradB[j]),
		})
	}
	for i := 0; i < batch; i++ {
		for k := 0; k < fIn; k++ {
			terms := make([]string, fOut)
			for j := 0; j < fOut; j++ {
				terms[j] = fmt.Sprintf("%s \\cdot %s", wrapNeg(f.GradOutput[i][j]), wrapNeg(f.Weights[j][k]))
			}
			view.GradXSteps = append(view.GradXSteps, expansionStep{
				LHS:    fmt.Sprintf("%d%d", i, k),
				Body:   strings.Join(terms, " + "),
				Result: fmtNum(f.GradX[i][k]),
			})
		}
	}

	tmpl, err := template.New("linear").Parse(linearTemplate)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	if err := tmpl.Execute(&sb, view); err != nil {
		return "", err
	}
	return sb.String(), nil
}

// rowList prints a row as comma separated values for prose
func rowList(row []float64) string {
	parts := make([]string, len(row))
	for i, v := range row {
		parts[i] = fmtNum(v)
	}
	return strings.Join(parts, ", ")
}

// wrapNeg parenthesizes negative numbers inside products and sums
func wrapNeg(v float64) string {
	if v < 0 {
		return "(" + fmtNum(v) + ")"
	}
	return fmtNum(v)
}

func matrixNode(g *gorgonia.ExprGraph, name string, vals [][]float64) *gorgonia.Node {
	rows := len(vals)
	cols := len(vals[0])
	return gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(rows, cols), gorgonia.WithName(name), gorgonia.WithValue(tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(flatten2(vals)))))
}

func flatten2(m [][]float64) []float64 {
	out := make([]float64, 0, len(m)*len(m[0]))
	for _, row := range m {
		out = append(out, row...)
	}
	return out
}

func compareFlat(name string, got, want []float64) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s: length mismatch: got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if !almostEqual(got[i], want[i]) {
			return fmt.Errorf("%s: element %d mismatch: got %v, want %v", name, i, got[i], want[i])
		}
	}
	return nil
}
