package gan_go

import (
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestAvgpoolForward(t *testing.T) {
	// Input 1x1x4x4, kernel 2x2, stride 2: output is 2x2 averages of quadrants
	data := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	want := []float64{
		(1 + 2 + 5 + 6) / 4.0, (3 + 4 + 7 + 8) / 4.0,
		(9 + 10 + 13 + 14) / 4.0, (11 + 12 + 15 + 16) / 4.0,
	}
	g := gorgonia.NewGraph()
	in := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(1, 1, 4, 4), gorgonia.WithName("avg_in"), gorgonia.WithValue(tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(data))))
	layer := &AvgpoolLayer{
		KernelHeight: 2,
		KernelWidth:  2,
		Padding:      []int{0, 0},
		Stride:       []int{2, 2},
	}
	out, err := layer.Fwd(1, in)
	if err != nil {
		t.Fatalf("Fwd error: %v", err)
	}
	vm := gorgonia.NewTapeMachine(g)
	defer vm.Close()
	if err := vm.RunAll(); err != nil {
		t.Fatalf("vm error: %v", err)
	}
	got := out.Value().Data().([]float64)
	for i := range want {
		checkFloat(t, "avgpool output", got[i], want[i], 1e-12)
	}
}
