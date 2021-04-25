package gan_go

import (
	"fmt"
	"image/color"
	"math/rand"

	"github.com/pkg/errors"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// NormRandDense Return reference to tensor.Dense filled with normally distributed float64 values in range [-inf;+inf] ([-maxF64;+maxF64 actually] actually)
//
// batchSize - Simply batch size
// n - Number of elements in each batch
// Resulting dense will have batchSize*n elements
//
func NormRandDense(batchSize, n int) *tensor.Dense {
	data := make([]float64, batchSize*n)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return tensor.New(tensor.WithShape(batchSize, n), tensor.WithBacking(data))
}

// NormRandDense Return reference to tensor.Dense filled with pseudo-random float64 values in range [0.0,1.0)
//
// batchSize - Simply batch size
// n - Number of elements in each batch
// Resulting dense will have batchSize*n elements
//
func UniformRandDense(batchSize, n int) *tensor.Dense {
	data := make([]float64, batchSize*n)
	for i := range data {
		data[i] = rand.Float64()
	}
	return tensor.New(tensor.WithShape(batchSize, n), tensor.WithBacking(data))
}

type ReferenceFunction func(float64) float64
type ArgumentFunction func() float64

func GenerateTrainingSet(numSamples int, xFunc ArgumentFunction, yFunc ReferenceFunction) (*TrainSet, error) {
	dataXAxis := make([]float64, numSamples)
	dataYAxis := make([]float64, numSamples)
	for i := range dataXAxis {
		dataXAxis[i] = xFunc()
		dataYAxis[i] = yFunc(dataXAxis[i])
	}
	inputTensor := tensor.New(tensor.WithShape(numSamples, 1), tensor.WithBacking(dataXAxis))
	outputTensor := tensor.New(tensor.WithShape(numSamples, 1), tensor.WithBacking(dataYAxis))
	hstack, err := inputTensor.Hstack(outputTensor)
	if err != nil {
		return nil, err
	}
	zeros := tensor.Ones(tensor.Float64, numSamples, 1)
	zeros.Zero()
	return &TrainSet{
		TrainData:  hstack,
		TrainLabel: zeros,
		DataLength: numSamples,
	}, nil
}

// SlicerOneStep Just iterator with step size = 1
type SlicerOneStep struct {
	StartIdx, EndIdx int
}

func (s SlicerOneStep) Start() int { return s.StartIdx }
func (s SlicerOneStep) End() int   { return s.EndIdx }
func (s SlicerOneStep) Step() int  { return 1 }

// PlotXY Plot chart for input y(x)
func PlotXY(x, y tensor.Tensor, fname string) error {
	if x.Dims() != 1 {
		return fmt.Errorf("X must have one dimension, but got %d", x.Dims())
	}
	if y.Dims() != 1 {
		return fmt.Errorf("Y(X) must have one dimension, but got %d", x.Dims())
	}
	if x.DataSize() != y.DataSize() {
		return fmt.Errorf("X and Y(X) must have same number of elements, but X has %d elements and Y(X) has %d elements", x.DataSize(), y.DataSize())
	}
	scatterData := make(plotter.XYs, x.DataSize())
	for i := 0; i < x.DataSize(); i++ {
		xval, err := x.At(i)
		if err != nil {
			return errors.Wrap(err, "Can't select X-value")
		}
		yval, err := y.At(i)
		if err != nil {
			return errors.Wrap(err, "Can't select Y(x)-value")
		}
		// Do no cast interfaces{} to any type when you are not sure about types
		scatterData[i].X = xval.(float64)
		scatterData[i].Y = yval.(float64)
	}
	scatter, err := plotter.NewScatter(scatterData)
	if err != nil {
		return errors.Wrap(err, "Can't init new scatter")
	}
	scatter.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
	p := plot.New()
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())
	p.Add(scatter)
	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, fname); err != nil {
		return errors.Wrap(err, "Can't save plot")
	}
	return nil
}

// GenerateTestSamples Generates test samples for provided arguments
//
// vmGenerator - tape machine used for GAN
// vmDiscriminator - tape machined used for Discriminator only
// inputGenerator - node for holding value of Generator's input
// inputDiscriminator - node for holding value of Discriminator's input
// graphValue - variable with access to Generator's output
// numSamples - how many sample generate
// batchSize - batch size basically
// n - number of elements in each batch
//
func GenerateTestSamples(vmGenerator, vmDiscriminator gorgonia.VM, inputGenerator, inputDiscriminator *gorgonia.Node, generatorOutValue gorgonia.Value, numSamples, batchSize, n int) (*tensor.Dense, error) {
	var testSamplesTensor *tensor.Dense

	for i := 0; i < numSamples; i++ {
		latent_space_samples := NormRandDense(batchSize, n)
		err := gorgonia.Let(inputGenerator, latent_space_samples)
		if err != nil {
			return nil, errors.Wrap(err, "Can't init input value")
		}
		err = vmGenerator.RunAll()
		if err != nil {
			return nil, errors.Wrap(err, "Can't run VM")
		}
		vmGenerator.Reset()
		tensorV := generatorOutValue.(*tensor.Dense)
		tensorVConcat, err := tensor.Concat(0, tensorV, tensorV)
		if err != nil {
			return nil, errors.Wrap(err, "Can't do concatenation")
		}
		err = gorgonia.Let(inputDiscriminator, tensorVConcat)
		if err != nil {
			panic(err)
		}
		err = vmDiscriminator.RunAll()
		if err != nil {
			panic(err)
		}
		vmDiscriminator.Reset()
		if i == 0 {
			testSamplesTensor = tensorV
		} else {
			newT, err := testSamplesTensor.Vstack(tensorV)
			if err != nil {
				panic(err)
			}
			testSamplesTensor = newT
		}
	}
	return testSamplesTensor, nil
}
