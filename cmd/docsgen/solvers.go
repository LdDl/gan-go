package main

import (
	_ "embed"
	"fmt"
	"math"
	"os"
	"strings"
	"text/template"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

//go:embed templates/solvers.md.tmpl
var solversTemplate string

// solversFixture Traces of three solvers over the same parameter and gradient sequence
type solversFixture struct {
	Layer        string      `json:"layer"`
	Theta0       []float64   `json:"theta0"`
	Gradients    [][]float64 `json:"gradients"`
	LearningRate float64     `json:"learning_rate"`
	Rho          float64     `json:"rho"`
	Beta1        float64     `json:"beta1"`
	Beta2        float64     `json:"beta2"`
	Eps          float64     `json:"eps"`
	Vanilla      [][]float64 `json:"vanilla_theta"`
	RMSCache     [][]float64 `json:"rmsprop_cache"`
	RMSTheta     [][]float64 `json:"rmsprop_theta"`
	AdamM        [][]float64 `json:"adam_m"`
	AdamV        [][]float64 `json:"adam_v"`
	AdamTheta    [][]float64 `json:"adam_theta"`
}

// valueGradPair Minimal ValueGrad implementation for driving solvers with hand-picked gradients
type valueGradPair struct {
	v gorgonia.Value
	g gorgonia.Value
}

func (p *valueGradPair) Value() gorgonia.Value         { return p.v }
func (p *valueGradPair) Grad() (gorgonia.Value, error) { return p.g, nil }

func generateSolvers() error {
	theta0 := []float64{1, -2}
	grads := [][]float64{
		{2, -4},
		{1, 2},
		{-3, 1},
	}
	eta := 0.1
	rho := 0.9
	beta1 := 0.9
	beta2 := 0.999
	eps := 1e-8
	n := len(theta0)
	steps := len(grads)

	// Manual vanilla SGD
	vanilla := [][]float64{append([]float64{}, theta0...)}
	for t := 0; t < steps; t++ {
		prev := vanilla[len(vanilla)-1]
		next := make([]float64, n)
		for k := 0; k < n; k++ {
			next[k] = prev[k] - eta*grads[t][k]
		}
		vanilla = append(vanilla, next)
	}

	// Manual RMSProp
	rmsTheta := [][]float64{append([]float64{}, theta0...)}
	rmsCache := [][]float64{}
	cache := make([]float64, n)
	for t := 0; t < steps; t++ {
		prev := rmsTheta[len(rmsTheta)-1]
		next := make([]float64, n)
		for k := 0; k < n; k++ {
			cache[k] = rho*cache[k] + (1-rho)*grads[t][k]*grads[t][k]
			next[k] = prev[k] - eta*grads[t][k]/math.Sqrt(cache[k]+eps)
		}
		rmsCache = append(rmsCache, append([]float64{}, cache...))
		rmsTheta = append(rmsTheta, next)
	}

	// Manual Adam
	adamTheta := [][]float64{append([]float64{}, theta0...)}
	adamM := [][]float64{}
	adamV := [][]float64{}
	m := make([]float64, n)
	v := make([]float64, n)
	for t := 0; t < steps; t++ {
		prev := adamTheta[len(adamTheta)-1]
		next := make([]float64, n)
		corr1 := 1 - math.Pow(beta1, float64(t+1))
		corr2 := 1 - math.Pow(beta2, float64(t+1))
		for k := 0; k < n; k++ {
			m[k] = beta1*m[k] + (1-beta1)*grads[t][k]
			v[k] = beta2*v[k] + (1-beta2)*grads[t][k]*grads[t][k]
			mHat := m[k] / corr1
			vHat := v[k] / corr2
			next[k] = prev[k] - eta*mHat/(math.Sqrt(vHat)+eps)
		}
		adamM = append(adamM, append([]float64{}, m...))
		adamV = append(adamV, append([]float64{}, v...))
		adamTheta = append(adamTheta, next)
	}

	if err := verifySolver("vanilla", gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(eta)), theta0, grads, vanilla); err != nil {
		return err
	}
	if err := verifySolver("rmsprop", gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(eta), gorgonia.WithRho(rho), gorgonia.WithEps(eps)), theta0, grads, rmsTheta); err != nil {
		return err
	}
	if err := verifySolver("adam", gorgonia.NewAdamSolver(gorgonia.WithLearnRate(eta), gorgonia.WithBeta1(beta1), gorgonia.WithBeta2(beta2), gorgonia.WithEps(eps)), theta0, grads, adamTheta); err != nil {
		return err
	}

	fixture := solversFixture{
		Layer:        "solvers",
		Theta0:       theta0,
		Gradients:    grads,
		LearningRate: eta,
		Rho:          rho,
		Beta1:        beta1,
		Beta2:        beta2,
		Eps:          eps,
		Vanilla:      vanilla,
		RMSCache:     rmsCache,
		RMSTheta:     rmsTheta,
		AdamM:        adamM,
		AdamV:        adamV,
		AdamTheta:    adamTheta,
	}
	if err := writeJSON("solvers", fixture); err != nil {
		return err
	}
	return spliceSolversSection(fixture)
}

// verifySolver drives the actual Gorgonia solver with the fixture gradients and compares every step
func verifySolver(name string, solver gorgonia.Solver, theta0 []float64, grads, want [][]float64) error {
	n := len(theta0)
	value := tensor.New(tensor.WithShape(n), tensor.WithBacking(append([]float64{}, theta0...)))
	grad := tensor.New(tensor.WithShape(n), tensor.WithBacking(make([]float64, n)))
	pair := &valueGradPair{v: value, g: grad}
	for t := range grads {
		copy(grad.Data().([]float64), grads[t])
		if err := solver.Step([]gorgonia.ValueGrad{pair}); err != nil {
			return fmt.Errorf("%s step %d: %w", name, t+1, err)
		}
		if err := compareFlat(fmt.Sprintf("%s theta after step %d", name, t+1), value.Data().([]float64), want[t+1]); err != nil {
			return err
		}
	}
	return nil
}

// spliceSolversSection injects the generated numeric section into docs/solvers.md between markers
func spliceSolversSection(f solversFixture) error {
	const path = "docs/solvers.md"
	const begin = "<!-- numeric:begin -->"
	const end = "<!-- numeric:end -->"
	src, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	content := string(src)
	numeric, err := solversNumericSection(f)
	if err != nil {
		return err
	}
	section := begin + "\n" + numeric + end
	if strings.Contains(content, begin) && strings.Contains(content, end) {
		head := content[:strings.Index(content, begin)]
		tail := content[strings.Index(content, end)+len(end):]
		content = head + section + tail
	} else {
		refIdx := strings.Index(content, "## References")
		if refIdx < 0 {
			return fmt.Errorf("no References section in %s", path)
		}
		content = content[:refIdx] + section + "\n\n" + content[refIdx:]
	}
	return os.WriteFile(path, []byte(content), 0644)
}

func vec(v []float64) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = fmtNum(x)
	}
	return "(" + strings.Join(parts, ",\\; ") + ")"
}

// solverRow One table row of the numeric traces
type solverRow struct {
	T                 int
	G, R, M, V, Theta string
}

// solversView View model for the solvers template: all inline numbers are derived from the fixture
type solversView struct {
	Theta0, G1, G2, G3     string
	Eta, Rho, Beta1, Beta2 string
	VanillaFirst           string
	RmsR1, RmsTheta1       string
	AdamM1, AdamV1         string
	AdamMhat, AdamVhat     string
	AdamTheta1             string
	VanillaRows            []solverRow
	RmsRows                []solverRow
	AdamRows               []solverRow
}

func solversNumericSection(f solversFixture) (string, error) {
	view := solversView{
		Theta0: vec(f.Theta0),
		G1:     vec(f.Gradients[0]),
		G2:     vec(f.Gradients[1]),
		G3:     vec(f.Gradients[2]),
		Eta:    fmtNum(f.LearningRate),
		Rho:    fmtNum(f.Rho),
		Beta1:  fmtNum(f.Beta1),
		Beta2:  fmtNum(f.Beta2),
	}
	g0 := f.Gradients[0][0]
	view.VanillaFirst = fmt.Sprintf("%s - %s \\cdot %s = %s", fmtNum(f.Theta0[0]), fmtNum(f.LearningRate), fmtNum(g0), fmtNum(f.Vanilla[1][0]))
	view.RmsR1 = fmt.Sprintf("%s \\cdot 0 + %s \\cdot %s^2 = %s", fmtNum(f.Rho), fmtNum(1-f.Rho), fmtNum(g0), fmtNum(f.RMSCache[0][0]))
	view.RmsTheta1 = fmt.Sprintf("%s - %s \\cdot %s / \\sqrt{%s} \\approx %s", fmtNum(f.Theta0[0]), fmtNum(f.LearningRate), fmtNum(g0), fmtNum(f.RMSCache[0][0]), fmtNum(f.RMSTheta[1][0]))
	view.AdamM1 = fmt.Sprintf("%s \\cdot %s = %s", fmtNum(1-f.Beta1), fmtNum(g0), fmtNum(f.AdamM[0][0]))
	view.AdamV1 = fmt.Sprintf("%s \\cdot %s = %s", fmtNum(1-f.Beta2), fmtNum(g0*g0), fmtNum(f.AdamV[0][0]))
	view.AdamMhat = fmt.Sprintf("%s/%s = %s", fmtNum(f.AdamM[0][0]), fmtNum(1-f.Beta1), fmtNum(f.AdamM[0][0]/(1-f.Beta1)))
	view.AdamVhat = fmt.Sprintf("%s/%s = %s", fmtNum(f.AdamV[0][0]), fmtNum(1-f.Beta2), fmtNum(f.AdamV[0][0]/(1-f.Beta2)))
	view.AdamTheta1 = fmt.Sprintf("%s - %s \\cdot %s/\\sqrt{%s} = %s", fmtNum(f.Theta0[0]), fmtNum(f.LearningRate), fmtNum(f.AdamM[0][0]/(1-f.Beta1)), fmtNum(f.AdamV[0][0]/(1-f.Beta2)), fmtNum(f.AdamTheta[1][0]))
	for t := 0; t < len(f.Gradients); t++ {
		view.VanillaRows = append(view.VanillaRows, solverRow{
			T:     t + 1,
			G:     vec(f.Gradients[t]),
			Theta: vec(f.Vanilla[t+1]),
		})
		view.RmsRows = append(view.RmsRows, solverRow{
			T:     t + 1,
			G:     vec(f.Gradients[t]),
			R:     vec(f.RMSCache[t]),
			Theta: vec(f.RMSTheta[t+1]),
		})
		view.AdamRows = append(view.AdamRows, solverRow{
			T:     t + 1,
			G:     vec(f.Gradients[t]),
			M:     vec(f.AdamM[t]),
			V:     vec(f.AdamV[t]),
			Theta: vec(f.AdamTheta[t+1]),
		})
	}
	tmpl, err := template.New("solvers").Parse(solversTemplate)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	if err := tmpl.Execute(&sb, view); err != nil {
		return "", err
	}
	return sb.String(), nil
}
