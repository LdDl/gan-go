package main

import (
	"fmt"
	"math/rand"

	gan "github.com/LdDl/gan-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	learning_rate = 0.01
	batchSize     = 1
	imgHeight     = 9
	imgWidth      = 8
	imgChannels   = 1
	numExamples   = 1000
	numOfEpochs   = 1
	classes       = 3
	imgShape      = []int{batchSize, imgChannels, imgHeight, imgWidth}

	x_image_label = tensor.New(tensor.WithShape(classes), tensor.WithBacking([]float64{1, 0, 0}))
	x_image       = tensor.New(tensor.WithShape(imgShape...), tensor.WithBacking([]float64{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 1, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0, 0,
	}))

	t_image_label = tensor.New(tensor.WithShape(classes), tensor.WithBacking([]float64{0, 1, 0}))
	t_image       = tensor.New(tensor.WithShape(imgShape...), tensor.WithBacking([]float64{
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
	}))

	o_image_label = tensor.New(tensor.WithShape(classes), tensor.WithBacking([]float64{0, 0, 1}))
	o_image       = tensor.New(tensor.WithShape(imgShape...), tensor.WithBacking([]float64{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 0, 1, 1, 1, 1, 0, 0,
	}))
)

func main() {
	rand.Seed(1337)

	/* Define Gorgonia's graph */
	cnnGraph := gorgonia.NewGraph()

	/* Define structure of neural network */
	simpleCNN := defineCNN(cnnGraph)

	/* Prepare tensor for input values */
	inputCNN := gorgonia.NewTensor(cnnGraph, gorgonia.Float64, 4, gorgonia.WithShape(imgShape...), gorgonia.WithName("discriminator_train_input"))
	err := simpleCNN.Fwd(inputCNN, batchSize)
	if err != nil {
		panic(err)
	}
	/* Prepare tensor for label values */
	targetCNN := gorgonia.NewTensor(cnnGraph, gorgonia.Float64, 1, gorgonia.WithShape(classes), gorgonia.WithName("discriminator_label"))

	/* Prepare variable for storing neural network's output */
	var cnnOut gorgonia.Value
	gorgonia.Read(simpleCNN.Out(), &cnnOut)

	/* Prepare cost node */
	cost, err := gan.MSELoss(simpleCNN.Out(), targetCNN)
	if err != nil {
		panic(err)
	}
	gorgonia.WithName("discriminator_loss")(cost)

	/* Define gradients */
	_, err = gorgonia.Grad(cost, simpleCNN.Learnables()...)
	if err != nil {
		panic(err)
	}

	/* Prepare variable for storing neural network's cost */
	var costOut gorgonia.Value
	gorgonia.Read(cost, &costOut)

	/* Define tape machine */
	tm := gorgonia.NewTapeMachine(cnnGraph, gorgonia.BindDualValues(simpleCNN.Learnables()...))
	defer tm.Close()

	/* Initialize solver for evaluation graph */
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate))

	// Run through all epochs
	for e := 0; e < numOfEpochs; e++ {
		// Run through each hardcoded sample
		for i := 0; i < numExamples; i++ {

			/* Train neural network to recognize char 'X' */
			// Prepare input
			err = gorgonia.Let(inputCNN, x_image)
			if err != nil {
				panic(err)
			}
			// Prepare training label
			err = gorgonia.Let(targetCNN, x_image_label)
			if err != nil {
				panic(err)
			}

			/* Run training step */
			err = tm.RunAll()
			if err != nil {
				panic(err)
			}
			err = solver.Step(gorgonia.NodesToValueGrads(simpleCNN.Learnables()))
			if err != nil {
				panic(err)
			}
			tm.Reset()

			/* Train neural network to recognize char 'T' */
			// Prepare input
			err = gorgonia.Let(inputCNN, t_image)
			if err != nil {
				panic(err)
			}
			// Prepare training label
			err = gorgonia.Let(targetCNN, t_image_label)
			if err != nil {
				panic(err)
			}

			/* Run training step */
			err = tm.RunAll()
			if err != nil {
				panic(err)
			}
			err = solver.Step(gorgonia.NodesToValueGrads(simpleCNN.Learnables()))
			if err != nil {
				panic(err)
			}
			tm.Reset()

			/* Train neural network to recognize char 'O' */
			// Prepare input
			err = gorgonia.Let(inputCNN, o_image)
			if err != nil {
				panic(err)
			}
			// Prepare training label
			err = gorgonia.Let(targetCNN, o_image_label)
			if err != nil {
				panic(err)
			}

			/* Run training step */
			err = tm.RunAll()
			if err != nil {
				panic(err)
			}
			err = solver.Step(gorgonia.NodesToValueGrads(simpleCNN.Learnables()))
			if err != nil {
				panic(err)
			}
			tm.Reset()
		}
	}

	/* Test neural network on both training data and noisy test data */

	/* 'X' char */
	err = gorgonia.Let(inputCNN, x_image)
	if err != nil {
		panic(err)
	}
	err = tm.RunAll()
	if err != nil {
		panic(err)
	}
	tm.Reset()
	fmt.Println("X => Should give [1, 0, 0]", cnnOut)

	/* 'X' char [with noise] */
	x_image_noisy, err := addNoiseToNoZeroValues(x_image)
	if err != nil {
		panic(err)
	}
	err = gorgonia.Let(inputCNN, x_image_noisy)
	if err != nil {
		panic(err)
	}
	err = tm.RunAll()
	if err != nil {
		panic(err)
	}
	tm.Reset()
	fmt.Println("\tnoisy X => Should give [1, 0, 0]", cnnOut)

	/* 'T' char */
	err = gorgonia.Let(inputCNN, t_image)
	if err != nil {
		panic(err)
	}
	err = tm.RunAll()
	if err != nil {
		panic(err)
	}
	tm.Reset()
	fmt.Println("T => Should give [0, 1, 0]", cnnOut)

	/* 'T' char [with noise] */
	t_image_noisy, err := addNoiseToNoZeroValues(t_image)
	if err != nil {
		panic(err)
	}
	err = gorgonia.Let(inputCNN, t_image_noisy)
	if err != nil {
		panic(err)
	}
	err = tm.RunAll()
	if err != nil {
		panic(err)
	}
	tm.Reset()
	fmt.Println("\tnoisy T => Should give [0, 1, 0]", cnnOut)

	/* 'O' char */
	err = gorgonia.Let(inputCNN, o_image)
	if err != nil {
		panic(err)
	}
	err = tm.RunAll()
	if err != nil {
		panic(err)
	}
	tm.Reset()
	fmt.Println("O => Should give [0, 0, 1]", cnnOut)

	/* 'O' char [with noise] */
	o_image_noisy, err := addNoiseToNoZeroValues(o_image)
	if err != nil {
		panic(err)
	}
	err = gorgonia.Let(inputCNN, o_image_noisy)
	if err != nil {
		panic(err)
	}
	err = tm.RunAll()
	if err != nil {
		panic(err)
	}
	tm.Reset()
	fmt.Println("\tnoisy O => Should give [0, 0, 1]", cnnOut)
}

func defineCNN(g *gorgonia.ExprGraph) *gan.DiscriminatorNet {
	/*
		input(9,8) => filters=5,size=3x3,conv(7,6) => filters=5,size=2x2,maxpool(3,3) => 5*flatten(9) => linear(3, 45)
	*/
	dis_shp0 := tensor.Shape{5, imgChannels, 3, 3}
	dis_w0 := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(dis_shp0...), gorgonia.WithName("discriminator_train_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	dis_shp1 := tensor.Shape{3, 45}
	dis_w1 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(dis_shp1...), gorgonia.WithName("discriminator_train_w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	discriminator := gan.Discriminator(
		[]*gan.Layer{
			{
				WeightNode:   dis_w0,
				BiasNode:     nil,
				Type:         gan.LayerConvolutional,
				Activation:   gan.Rectify,
				KernelHeight: 3,
				KernelWidth:  3,
				Padding:      []int{0, 0},
				Stride:       []int{1, 1},
				Dilation:     []int{1, 1},
			},
			{
				Type:        gan.LayerDropout,
				Probability: 0.3,
			},
			{
				Type:         gan.LayerMaxpool,
				Activation:   gan.NoActivation,
				KernelHeight: 2,
				KernelWidth:  2,
				Padding:      []int{0, 0},
				Stride:       []int{2, 2},
			},
			{
				Type:       gan.LayerFlatten,
				Activation: gan.NoActivation,
			},
			{
				WeightNode: dis_w1,
				BiasNode:   nil,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
		}...,
	)
	return discriminator
}

func addNoiseToNoZeroValues(t *tensor.Dense) (*tensor.Dense, error) {
	numElements := t.Shape().TotalSize()
	newData := make([]float64, numElements)
	for i := 0; i < numElements; i++ {
		// Random value in [0; 0.2)
		newData[i] = rand.Float64() / 5.0
	}
	newTensor := tensor.New(tensor.WithShape(t.Shape()...), tensor.WithBacking(newData))
	return t.Add(newTensor)
}
