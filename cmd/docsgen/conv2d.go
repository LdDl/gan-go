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

//go:embed templates/conv2d.md.tmpl
var conv2dTemplate string

// conv2dFixture Payload of the convolutional layer example
type conv2dFixture struct {
	Layer    string          `json:"layer"`
	Channels int             `json:"channels"`
	Filters  int             `json:"filters"`
	KernelH  int             `json:"kernel_h"`
	KernelW  int             `json:"kernel_w"`
	Padding  []int           `json:"padding"`
	Stride   []int           `json:"stride"`
	Input    [][][]float64   `json:"input"`
	Padded   [][][]float64   `json:"padded"`
	Kernels  [][][][]float64 `json:"kernels"`
	// PartialOutputs[o][c] is the per-channel response map of filter o:
	// the channel slice applied alone, before summing over channels
	PartialOutputs [][][][]float64 `json:"partial_outputs"`
	Output         [][][]float64   `json:"output"`
	GradOutput     [][][]float64   `json:"grad_output"`
	GradW          [][][][]float64 `json:"grad_kernels"`
	GradX          [][][]float64   `json:"grad_input"`
}

// Fixture parameters: an RGB-like image of 13x9 (non-square), a non-square kernel 3x5,
// padding, a stride larger than one and two filters. Values are small integers
// produced by deterministic formulas, so the fixture is reproducible and hand checkable.
const (
	convChannels = 3
	convHeight   = 13
	convWidth    = 9
	convFilters  = 2
	convKernelH  = 3
	convKernelW  = 5
)

var (
	convPadding = []int{1, 1}
	convStride  = []int{2, 2}
)

func convInput() [][][]float64 {
	out := make([][][]float64, convChannels)
	for c := 0; c < convChannels; c++ {
		out[c] = make([][]float64, convHeight)
		for i := 0; i < convHeight; i++ {
			out[c][i] = make([]float64, convWidth)
			for j := 0; j < convWidth; j++ {
				out[c][i][j] = float64((i*convWidth+j+7*c)%5 - 1)
			}
		}
	}
	return out
}

func convKernels() [][][][]float64 {
	out := make([][][][]float64, convFilters)
	for o := 0; o < convFilters; o++ {
		out[o] = make([][][]float64, convChannels)
		for c := 0; c < convChannels; c++ {
			out[o][c] = make([][]float64, convKernelH)
			for m := 0; m < convKernelH; m++ {
				out[o][c][m] = make([]float64, convKernelW)
				for n := 0; n < convKernelW; n++ {
					out[o][c][m][n] = float64((m*convKernelW+n+4*c+5*o)%3 - 1)
				}
			}
		}
	}
	return out
}

func convDelta(outH, outW int) [][][]float64 {
	out := make([][][]float64, convFilters)
	for o := 0; o < convFilters; o++ {
		out[o] = make([][]float64, outH)
		for i := 0; i < outH; i++ {
			out[o][i] = make([]float64, outW)
			for j := 0; j < outW; j++ {
				out[o][i][j] = float64((i*outW+j+2*o)%4 - 2)
			}
		}
	}
	return out
}

func generateConv2D() error {
	input := convInput()
	kernels := convKernels()

	outH := (convHeight+2*convPadding[0]-convKernelH)/convStride[0] + 1
	outW := (convWidth+2*convPadding[1]-convKernelW)/convStride[1] + 1
	delta := convDelta(outH, outW)
	padded := padInput(input, convPadding[0], convPadding[1])

	// Manual forward: per-channel partial responses first, the output is their sum over channels
	partial := make([][][][]float64, convFilters)
	output := make([][][]float64, convFilters)
	for o := 0; o < convFilters; o++ {
		partial[o] = make([][][]float64, convChannels)
		for c := 0; c < convChannels; c++ {
			partial[o][c] = make([][]float64, outH)
			for i := 0; i < outH; i++ {
				partial[o][c][i] = make([]float64, outW)
				for j := 0; j < outW; j++ {
					s := 0.0
					for m := 0; m < convKernelH; m++ {
						for n := 0; n < convKernelW; n++ {
							s += padded[c][i*convStride[0]+m][j*convStride[1]+n] * kernels[o][c][m][n]
						}
					}
					partial[o][c][i][j] = s
				}
			}
		}
		output[o] = make([][]float64, outH)
		for i := 0; i < outH; i++ {
			output[o][i] = make([]float64, outW)
			for j := 0; j < outW; j++ {
				for c := 0; c < convChannels; c++ {
					output[o][i][j] += partial[o][c][i][j]
				}
			}
		}
	}

	// Manual backward: kernel gradients
	gradW := make([][][][]float64, convFilters)
	for o := 0; o < convFilters; o++ {
		gradW[o] = make([][][]float64, convChannels)
		for c := 0; c < convChannels; c++ {
			gradW[o][c] = make([][]float64, convKernelH)
			for m := 0; m < convKernelH; m++ {
				gradW[o][c][m] = make([]float64, convKernelW)
				for n := 0; n < convKernelW; n++ {
					s := 0.0
					for i := 0; i < outH; i++ {
						for j := 0; j < outW; j++ {
							s += delta[o][i][j] * padded[c][i*convStride[0]+m][j*convStride[1]+n]
						}
					}
					gradW[o][c][m][n] = s
				}
			}
		}
	}

	// Manual backward: input gradients via scatter over the padded canvas, then crop the padding ring
	gradPadded := make([][][]float64, convChannels)
	for c := 0; c < convChannels; c++ {
		gradPadded[c] = make([][]float64, convHeight+2*convPadding[0])
		for p := range gradPadded[c] {
			gradPadded[c][p] = make([]float64, convWidth+2*convPadding[1])
		}
	}
	for o := 0; o < convFilters; o++ {
		for i := 0; i < outH; i++ {
			for j := 0; j < outW; j++ {
				for c := 0; c < convChannels; c++ {
					for m := 0; m < convKernelH; m++ {
						for n := 0; n < convKernelW; n++ {
							gradPadded[c][i*convStride[0]+m][j*convStride[1]+n] += delta[o][i][j] * kernels[o][c][m][n]
						}
					}
				}
			}
		}
	}
	gradX := make([][][]float64, convChannels)
	for c := 0; c < convChannels; c++ {
		gradX[c] = make([][]float64, convHeight)
		for p := 0; p < convHeight; p++ {
			gradX[c][p] = make([]float64, convWidth)
			copy(gradX[c][p], gradPadded[c][p+convPadding[0]][convPadding[1]:convPadding[1]+convWidth])
		}
	}

	if err := verifyConv2D(input, kernels, delta, convPadding, convStride, output, gradW, gradX); err != nil {
		return err
	}

	fixture := conv2dFixture{
		Layer:          "conv2d",
		Channels:       convChannels,
		Filters:        convFilters,
		KernelH:        convKernelH,
		KernelW:        convKernelW,
		Padding:        convPadding,
		Stride:         convStride,
		Input:          input,
		Padded:         padded,
		Kernels:        kernels,
		PartialOutputs: partial,
		Output:         output,
		GradOutput:     delta,
		GradW:          gradW,
		GradX:          gradX,
	}
	if err := writeJSON("conv2d", fixture); err != nil {
		return err
	}
	content, err := conv2dMarkdown(fixture)
	if err != nil {
		return err
	}
	return writeMarkdown("conv2d", content)
}

func padInput(input [][][]float64, ph, pw int) [][][]float64 {
	channels := len(input)
	height := len(input[0])
	width := len(input[0][0])
	out := make([][][]float64, channels)
	for c := 0; c < channels; c++ {
		out[c] = make([][]float64, height+2*ph)
		for p := range out[c] {
			out[c][p] = make([]float64, width+2*pw)
		}
		for i := 0; i < height; i++ {
			for j := 0; j < width; j++ {
				out[c][i+ph][j+pw] = input[c][i][j]
			}
		}
	}
	return out
}

// verifyConv2D replays the fixture through Conv2DLayer and Gorgonia gradients
func verifyConv2D(input [][][]float64, kernels [][][][]float64, delta [][][]float64, padding, stride []int, wantOut [][][]float64, wantGW [][][][]float64, wantGX [][][]float64) error {
	channels := len(input)
	height := len(input[0])
	width := len(input[0][0])
	filters := len(kernels)
	kh := len(kernels[0][0])
	kw := len(kernels[0][0][0])

	g := gorgonia.NewGraph()
	xNode := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(1, channels, height, width), gorgonia.WithName("x"), gorgonia.WithValue(tensor.New(tensor.WithShape(1, channels, height, width), tensor.WithBacking(flatten3(input)))))
	wNode := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(filters, channels, kh, kw), gorgonia.WithName("w"), gorgonia.WithValue(tensor.New(tensor.WithShape(filters, channels, kh, kw), tensor.WithBacking(flatten4(kernels)))))
	layer := &gan.Conv2DLayer{
		WeightNode:   wNode,
		Activation:   gan.NoActivation,
		KernelHeight: kh,
		KernelWidth:  kw,
		Padding:      padding,
		Stride:       stride,
		Dilation:     []int{1, 1},
	}
	out, err := layer.Fwd(xNode)
	if err != nil {
		return err
	}
	outH := len(wantOut[0])
	outW := len(wantOut[0][0])
	deltaNode := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(1, filters, outH, outW), gorgonia.WithName("delta"), gorgonia.WithValue(tensor.New(tensor.WithShape(1, filters, outH, outW), tensor.WithBacking(flatten3(delta)))))
	weighted, err := gorgonia.HadamardProd(out, deltaNode)
	if err != nil {
		return err
	}
	cost, err := gorgonia.Sum(weighted)
	if err != nil {
		return err
	}
	learnables := gorgonia.Nodes{wNode, xNode}
	if _, err := gorgonia.Grad(cost, learnables...); err != nil {
		return err
	}
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(learnables...))
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		return err
	}
	if err := compareFlat("conv2d forward", out.Value().Data().([]float64), flatten3(wantOut)); err != nil {
		return err
	}
	gw, err := wNode.Grad()
	if err != nil {
		return err
	}
	if err := compareFlat("conv2d dW", gw.Data().([]float64), flatten4(wantGW)); err != nil {
		return err
	}
	gx, err := xNode.Grad()
	if err != nil {
		return err
	}
	return compareFlat("conv2d dx", gx.Data().([]float64), flatten3(wantGX))
}

func flatten3(m [][][]float64) []float64 {
	out := []float64{}
	for _, plane := range m {
		out = append(out, flatten2(plane)...)
	}
	return out
}

func flatten4(m [][][][]float64) []float64 {
	out := []float64{}
	for _, v := range m {
		out = append(out, flatten3(v)...)
	}
	return out
}

// patch extracts a kh x kw window of a padded channel at output position (i, j)
func patch(padded [][]float64, i, j, kh, kw, sh, sw int) [][]float64 {
	out := make([][]float64, kh)
	for m := 0; m < kh; m++ {
		out[m] = make([]float64, kw)
		for n := 0; n < kw; n++ {
			out[m][n] = padded[i*sh+m][j*sw+n]
		}
	}
	return out
}

// taggedTex A LaTeX fragment with its equation number and an optional name
type taggedTex struct {
	Name   string
	Index  int
	Tex    string
	Parts  string
	Slices string
	Tag    int
}

// positionView One fully expanded window position of the forward pass
type positionView struct {
	I, J               int
	StartRow, StartCol int
	Channels           []expansionChannel
	Total              string
}

type expansionChannel struct {
	Name     string
	PatchTex string
	Body     string
	Result   string
}

// conv2dView View model for the conv2d template
type conv2dView struct {
	Channels, Filters            int
	Height, Width                int
	KernelH, KernelW             int
	PadH, PadW, StrideH, StrideW int
	PaddedH, PaddedW, OutH, OutW int
	InputChannels                []taggedTex
	FilterBlocks                 []taggedTex
	PaddedR                      string
	PaddedTag                    int
	OutSizeTag                   int
	Positions                    []positionView
	PartialMaps                  []taggedTex
	Y0, Y1                       string
	Y0Tag, Y1Tag                 int
	Delta0, Delta1               string
	DeltaTag                     int
	DWAlignedBody                string
	DWAlignedTag                 int
	GradWBlocks                  []taggedTex
	PixRow, PixCol               int
	PixPadRow, PixPadCol         int
	PixBody, PixResult           string
	PixTag                       int
	GradXBlocks                  []taggedTex
}

func conv2dMarkdown(f conv2dFixture) (string, error) {
	outH := len(f.Output[0])
	outW := len(f.Output[0][0])
	height := len(f.Input[0])
	width := len(f.Input[0][0])
	channelNames := []string{"R", "G", "B"}

	view := conv2dView{
		Channels: f.Channels,
		Filters:  f.Filters,
		Height:   height,
		Width:    width,
		KernelH:  f.KernelH,
		KernelW:  f.KernelW,
		PadH:     f.Padding[0],
		PadW:     f.Padding[1],
		StrideH:  f.Stride[0],
		StrideW:  f.Stride[1],
		PaddedH:  height + 2*f.Padding[0],
		PaddedW:  width + 2*f.Padding[1],
		OutH:     outH,
		OutW:     outW,
		PaddedR:  texMatrix(f.Padded[0]),
		Y0:       texMatrix(f.Output[0]),
		Y1:       texMatrix(f.Output[1]),
		Delta0:   texMatrix(f.GradOutput[0]),
		Delta1:   texMatrix(f.GradOutput[1]),
	}

	eq := 1
	for c := 0; c < f.Channels; c++ {
		view.InputChannels = append(view.InputChannels, taggedTex{Name: channelNames[c], Tex: texMatrix(f.Input[c]), Tag: eq})
		eq++
	}
	for o := 0; o < f.Filters; o++ {
		parts := make([]string, f.Channels)
		for c := 0; c < f.Channels; c++ {
			parts[c] = fmt.Sprintf("k^{(%d,%s)} = %s", o, channelNames[c], texMatrix(f.Kernels[o][c]))
		}
		view.FilterBlocks = append(view.FilterBlocks, taggedTex{Index: o, Slices: strings.Join(parts, " \\qquad "), Tag: eq})
		eq++
	}
	view.PaddedTag = eq
	eq++
	view.OutSizeTag = eq
	eq++

	for _, pos := range [][2]int{{0, 0}, {2, 1}} {
		i, j := pos[0], pos[1]
		pv := positionView{
			I:        i,
			J:        j,
			StartRow: i * f.Stride[0],
			StartCol: j * f.Stride[1],
		}
		total := 0.0
		for c := 0; c < f.Channels; c++ {
			p := patch(f.Padded[c], i, j, f.KernelH, f.KernelW, f.Stride[0], f.Stride[1])
			s := 0.0
			terms := []string{}
			for m := 0; m < f.KernelH; m++ {
				for n := 0; n < f.KernelW; n++ {
					prod := p[m][n] * f.Kernels[0][c][m][n]
					s += prod
					if p[m][n] != 0 && f.Kernels[0][c][m][n] != 0 {
						terms = append(terms, fmt.Sprintf("%s \\cdot %s", wrapNeg(p[m][n]), wrapNeg(f.Kernels[0][c][m][n])))
					}
				}
			}
			total += s
			if len(terms) == 0 {
				terms = append(terms, "0")
			}
			pv.Channels = append(pv.Channels, expansionChannel{
				Name:     channelNames[c],
				PatchTex: texMatrix(p),
				Body:     strings.Join(terms, " + "),
				Result:   fmtNum(s),
			})
		}
		pv.Total = fmtNum(total)
		view.Positions = append(view.Positions, pv)
	}

	for c := 0; c < f.Channels; c++ {
		view.PartialMaps = append(view.PartialMaps, taggedTex{Name: channelNames[c], Tex: texMatrix(f.PartialOutputs[0][c]), Tag: eq})
		eq++
	}
	view.Y0Tag = eq
	eq++
	view.Y1Tag = eq
	eq++
	view.DeltaTag = eq
	eq++

	var dw strings.Builder
	for i := 0; i < outH; i++ {
		terms := []string{}
		for j := 0; j < outW; j++ {
			d := f.GradOutput[0][i][j]
			x := f.Padded[0][i*f.Stride[0]][j*f.Stride[1]]
			if d != 0 && x != 0 {
				terms = append(terms, fmt.Sprintf("%s \\cdot %s", wrapNeg(d), wrapNeg(x)))
			}
		}
		if len(terms) == 0 {
			terms = append(terms, "0")
		}
		if i > 0 {
			dw.WriteString("&+ ")
		} else {
			dw.WriteString("&\\phantom{+} ")
		}
		dw.WriteString(strings.Join(terms, " + "))
		if i < outH-1 {
			dw.WriteString(" \\\\\n")
		}
	}
	fmt.Fprintf(&dw, " \\\\\n&= %s", fmtNum(f.GradW[0][0][0][0]))
	view.DWAlignedBody = dw.String()
	view.DWAlignedTag = eq
	eq++

	for o := 0; o < f.Filters; o++ {
		parts := make([]string, f.Channels)
		for c := 0; c < f.Channels; c++ {
			parts[c] = fmt.Sprintf("\\frac{\\partial L}{\\partial k^{(%d,%s)}} = %s", o, channelNames[c], texMatrix(f.GradW[o][c]))
		}
		view.GradWBlocks = append(view.GradWBlocks, taggedTex{Parts: strings.Join(parts, " \\qquad "), Tag: eq})
		eq++
	}

	view.PixRow, view.PixCol = 1, 3
	view.PixPadRow = view.PixRow + f.Padding[0]
	view.PixPadCol = view.PixCol + f.Padding[1]
	pixTerms := []string{}
	pixSum := 0.0
	for o := 0; o < f.Filters; o++ {
		for i := 0; i < outH; i++ {
			for j := 0; j < outW; j++ {
				m := view.PixPadRow - i*f.Stride[0]
				n := view.PixPadCol - j*f.Stride[1]
				if m >= 0 && m < f.KernelH && n >= 0 && n < f.KernelW {
					prod := f.GradOutput[o][i][j] * f.Kernels[o][0][m][n]
					pixSum += prod
					if f.GradOutput[o][i][j] != 0 && f.Kernels[o][0][m][n] != 0 {
						pixTerms = append(pixTerms, fmt.Sprintf("\\underbrace{%s \\cdot %s}_{\\delta^{(%d)}_{%d%d} k^{(%d,R)}_{%d%d}}", wrapNeg(f.GradOutput[o][i][j]), wrapNeg(f.Kernels[o][0][m][n]), o, i, j, o, m, n))
					}
				}
			}
		}
	}
	view.PixBody = strings.Join(pixTerms, " + ")
	view.PixResult = fmtNum(pixSum)
	view.PixTag = eq
	eq++

	for c := 0; c < f.Channels; c++ {
		view.GradXBlocks = append(view.GradXBlocks, taggedTex{Name: channelNames[c], Tex: texMatrix(f.GradX[c]), Tag: eq})
		eq++
	}

	tmpl, err := template.New("conv2d").Parse(conv2dTemplate)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	if err := tmpl.Execute(&sb, view); err != nil {
		return "", err
	}
	return strings.TrimSuffix(sb.String(), "\n"), nil
}
