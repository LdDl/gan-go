package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	gan "github.com/LdDl/gan-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	learning_rate = 0.001
	batchSize     = 1
	imgHeight     = 10
	imgWidth      = 9
	imgChannels   = 1
	imgShape      = []int{batchSize, imgChannels, imgHeight, imgWidth}
	faceData      = []float64{
		0, 1, 1, 0, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 0, 1, 1, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0,
		1, 1, 0, 0, 0, 0, 0, 1, 1,
		0, 1, 1, 1, 0, 1, 1, 1, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0,
	}
	smiley_face_image = tensor.New(tensor.WithShape(imgShape...), tensor.WithBacking(faceData))

	latentSpaceSize = 250
	latentShape     = []int{batchSize, imgChannels, latentSpaceSize, latentSpaceSize}
	numEpoches      = 1500
	numTestSamples  = 1
	evalPrint       = 50
)

func genSyntheticData(numSamples int) *gan.TrainSet {
	fmt.Println("Actual smiley face:")
	for x := 0; x < imgHeight; x++ {
		fmt.Printf("\t")
		for y := 0; y < imgWidth; y++ {
			r := math.Round(faceData[x*imgWidth+y])
			char := "x"
			if r == 0 {
				char = " "
			}
			fmt.Printf("%s ", char)
		}
		fmt.Println()
	}

	labels := tensor.Ones(tensor.Float64, numSamples, 1)
	return &gan.TrainSet{
		TrainData:  smiley_face_image,
		TrainLabel: labels,
		DataLength: numSamples,
	}
}

func main() {
	// Initialize seed with constant value to reproduce results
	rand.Seed(1337)

	// Prepare synthetic data
	trainDataLength := 1
	trainSet := genSyntheticData(trainDataLength)

	// Define graph for GAN feedforward and Generator training
	ganGraph := gorgonia.NewGraph()
	// Define graph for Discriminator training
	trainDiscriminatorGraph := gorgonia.NewGraph()

	// Define Generator on GAN's evaluation graph
	definedGenerator := defineGenerator(ganGraph)
	// Initialize Generator feedforward
	inputGenerator := gorgonia.NewTensor(ganGraph, gorgonia.Float64, 4, gorgonia.WithShape(latentShape...), gorgonia.WithName("generator_input"))
	err := definedGenerator.Fwd(inputGenerator, batchSize)
	if err != nil {
		panic(err)
	}

	// Define Discriminator on its own evaluation graph
	discriminatorTrain := defineDiscriminator(trainDiscriminatorGraph)
	// Initialize Discriminator feedforward
	inputDiscriminatorTrain := gorgonia.NewTensor(trainDiscriminatorGraph, gorgonia.Float64, 4, gorgonia.WithShape(2*batchSize, imgChannels, imgHeight, imgWidth), gorgonia.WithName("discriminator_train_input"))
	err = discriminatorTrain.Fwd(inputDiscriminatorTrain, 2*batchSize)
	if err != nil {
		panic(err)
	}

	// Define GAN on the same evaluation graph as Generator has been defined
	definedGAN, err := gan.NewGAN(ganGraph, definedGenerator, discriminatorTrain)
	if err != nil {
		panic(err)
	}
	err = definedGAN.Fwd(batchSize)
	if err != nil {
		panic(err)
	}

	/* Define variables for reading evaluation graphs' (both GAN and Discriminator in training mode) output */
	// GAN Generator output
	var generatedSamples gorgonia.Value
	gorgonia.Read(definedGAN.GeneratorOut(), &generatedSamples)

	// GAN overall output (Discriminator output actually)
	var outputDiscriminator gorgonia.Value
	gorgonia.Read(definedGAN.Out(), &outputDiscriminator)

	// Discriminator output in training mode
	var outputDiscriminatorTrain gorgonia.Value
	gorgonia.Read(discriminatorTrain.Out(), &outputDiscriminatorTrain)

	// Initialize machine for GAN evaluation graph
	tmGenerator := gorgonia.NewTapeMachine(ganGraph)
	defer tmGenerator.Close()

	// Define loss function for GAN as
	targetDiscriminatorGAN := gorgonia.NewTensor(ganGraph, gorgonia.Float64, definedGAN.Out().Dims(), gorgonia.WithShape(definedGAN.Out().Shape()...), gorgonia.WithName("gan_discriminator_target"))
	cost, err := gan.BinaryCrossEntropyLoss(definedGAN.Out(), targetDiscriminatorGAN)
	if err != nil {
		panic(err)
	}
	gorgonia.WithName("gan_discriminator_loss")(cost)
	// Define gradients for GAN
	_, err = gorgonia.Grad(cost, definedGAN.Learnables()...)
	if err != nil {
		panic(err)
	}

	// Define loss function for Discriminator in training mode as
	targetDiscriminatorTrain := gorgonia.NewTensor(trainDiscriminatorGraph, gorgonia.Float64, 2, gorgonia.WithShape(2*batchSize, 1), gorgonia.WithName("discriminator_target"))
	costDiscriminatorTrain, err := gan.BinaryCrossEntropyLoss(discriminatorTrain.Out(), targetDiscriminatorTrain)
	if err != nil {
		panic(err)
	}
	gorgonia.WithName("discriminator_loss")(costDiscriminatorTrain)
	// Define gradients for Discriminator in training mode
	_, err = gorgonia.Grad(costDiscriminatorTrain, discriminatorTrain.Learnables()...)
	if err != nil {
		panic(err)
	}

	/* Read costs nodes into variable for further outputting */
	// GAN cost
	var costValGAN gorgonia.Value
	gorgonia.Read(cost, &costValGAN)

	// Discriminator [in training mode] cost
	var costValDiscriminatorTrain gorgonia.Value
	gorgonia.Read(costDiscriminatorTrain, &costValDiscriminatorTrain)

	// Tape machine for GAN evaluation graph
	tm := gorgonia.NewTapeMachine(ganGraph, gorgonia.BindDualValues(definedGAN.Learnables()...))
	defer tm.Close()
	// Solver for GAN evaluation graph
	solverGAN := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate))
	// Tape machine for Discriminator [in training mode] evaluation graph
	tmDisTrain := gorgonia.NewTapeMachine(trainDiscriminatorGraph, gorgonia.BindDualValues(discriminatorTrain.Learnables()...))
	defer tmDisTrain.Close()
	// Solver for Discriminator [in training mode] evaluation graph
	solverDiscriminatorTrain := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate))

	/* Training process */
	// Define number of batches as
	// baches_num = train_data_num / batch_size
	batches := int(trainDataLength / batchSize)
	// Looping process
	st := time.Now()
	for epoch := 0; epoch < numEpoches; epoch++ {
		// Iterate through batches
		for b := 0; b < batches; b++ {
			start := b * batchSize
			end := start + batchSize
			if start >= trainDataLength {
				break
			}
			if end > trainDataLength {
				end = trainDataLength
			}
			var xVal tensor.Tensor

			/* Batch real data */
			xVal = trainSet.TrainData
			real_samples_labels := tensor.Ones(tensor.Float64, batchSize, 1)
			latentSpaceSamples := gan.NormRandDense(batchSize, imgChannels*latentSpaceSize*latentSpaceSize)
			err := latentSpaceSamples.Reshape(latentShape...)
			if err != nil {
				panic(err)
			}
			err = gorgonia.Let(inputGenerator, latentSpaceSamples)
			if err != nil {
				panic(err)
			}
			// Do step on evaluation graph for obtaining 'generatedSamples' (Generator output)
			err = tmGenerator.RunAll()
			if err != nil {
				panic(err)
			}
			tmGenerator.Reset()
			// Assume that Generator generates wrong data, and label its output as zero
			generated_samples_labels := tensor.Ones(tensor.Float64, batchSize, 1)
			generated_samples_labels.Zero()
			// Concat real and fake input data
			all_samples, err := tensor.Concat(0, xVal, generatedSamples.(tensor.Tensor))
			if err != nil {
				fmt.Println(xVal.Shape(), generatedSamples.(tensor.Tensor).Shape())
				panic(err)
			}
			// Concat real and fake output labels
			all_samples_labels, err := tensor.Concat(0, real_samples_labels, generated_samples_labels)
			if err != nil {
				panic(err)
			}

			// Train discriminator on real data all_samples[0] (it's xVal and real_samples_labels respectively)
			err = gorgonia.Let(inputDiscriminatorTrain, all_samples)
			if err != nil {
				panic(err)
			}
			err = gorgonia.Let(targetDiscriminatorTrain, all_samples_labels)
			if err != nil {
				panic(err)
			}
			// Do training step for Discriminator in training mode
			err = tmDisTrain.RunAll()
			if err != nil {
				panic(err)
			}
			err = solverDiscriminatorTrain.Step(gorgonia.NodesToValueGrads(discriminatorTrain.Learnables()))
			if err != nil {
				panic(err)
			}
			tmDisTrain.Reset()
			latentSpaceSamplesGenerated := gan.NormRandDense(batchSize, imgChannels*latentSpaceSize*latentSpaceSize)
			err = latentSpaceSamplesGenerated.Reshape(latentShape...)
			if err != nil {
				panic(err)
			}
			err = gorgonia.Let(inputGenerator, latentSpaceSamplesGenerated)
			if err != nil {
				panic(err)
			}
			err = gorgonia.Let(targetDiscriminatorGAN, real_samples_labels)
			if err != nil {
				panic(err)
			}
			// Do training step for Generator
			err = tm.RunAll()
			if err != nil {
				panic(err)
			}
			err = solverGAN.Step(gorgonia.NodesToValueGrads(definedGAN.GeneratorLearnables()))
			if err != nil {
				panic(err)
			}
			tm.Reset()
			if epoch%evalPrint == 0 && b == batchSize-1 {
				fmt.Printf("Epoch %d:\n", epoch)
				fmt.Printf("\tDiscriminator's loss: %v\n", costValDiscriminatorTrain)
				fmt.Printf("\tGenerator's loss: %v\n", costValGAN)
				fmt.Printf("\tTaken time: %v\n", time.Since(st))
				st = time.Now()
				testSamplesTensor, err := gan.GenerateUniformTestSamples(tmGenerator, tmDisTrain, inputGenerator, inputDiscriminatorTrain, generatedSamples, numTestSamples, batchSize, imgChannels*latentSpaceSize*latentSpaceSize, latentShape)
				if err != nil {
					panic(err)
				}
				testData := testSamplesTensor.Materialize().Data().([]float64)
				for x := 0; x < imgHeight; x++ {
					fmt.Printf("\t")
					for y := 0; y < imgWidth; y++ {
						r := math.Round(testData[x*imgWidth+y])
						char := "x"
						if r < 0.5 {
							char = " "
						}
						fmt.Printf("%s ", char)
					}
					fmt.Println()
				}
			}
		}
	}

	// Final test of Generator
	fmt.Println("Start testing generator after final epoch")

	testSamplesTensor, err := gan.GenerateUniformTestSamples(tmGenerator, tmDisTrain, inputGenerator, inputDiscriminatorTrain, generatedSamples, numTestSamples, batchSize, imgChannels*latentSpaceSize*latentSpaceSize, latentShape)
	if err != nil {
		panic(err)
	}
	testData := testSamplesTensor.Materialize().Data().([]float64)
	for x := 0; x < imgHeight; x++ {
		fmt.Printf("\t")
		for y := 0; y < imgWidth; y++ {
			r := math.Round(testData[x*imgWidth+y])
			char := "x"
			if r < 0.5 {
				char = " "
			}
			fmt.Printf("%s ", char)
		}
		fmt.Println()
	}

	/* 'X' char */

	faceData = []float64{
		// Face
		0, 1, 1, 0, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 0, 1, 1, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0,
		1, 1, 0, 0, 0, 0, 0, 1, 1,
		0, 1, 1, 1, 0, 1, 1, 1, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0,

		// Not a face
		0, 0, 0, 0, 0, 0, 0, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 1, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1, 1, 0, 0,
	}
	smiley_face_image_x2 := tensor.New(tensor.WithShape(2*batchSize, imgChannels, imgHeight, imgWidth), tensor.WithBacking(faceData))
	err = gorgonia.Let(inputDiscriminatorTrain, smiley_face_image_x2)
	if err != nil {
		panic(err)
	}
	err = tmDisTrain.RunAll()
	if err != nil {
		panic(err)
	}
	tmDisTrain.Reset()
	fmt.Println("X => Should give [1, 0, 0]", outputDiscriminatorTrain)
}

func defineDiscriminator(g *gorgonia.ExprGraph) *gan.DiscriminatorNet {
	/*
		input(10,9) => filters=12,size=3x3,conv(8,7) => filters=12,size=2x2,maxpool(4,3) => 12*flatten(4*3) => linear(1, 12*4*3)
	*/
	dis_shp0 := tensor.Shape{12, imgChannels, 3, 3}
	dis_w0 := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(dis_shp0...), gorgonia.WithName("discriminator_train_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	dis_shp1 := tensor.Shape{1, 12 * 4 * 3}
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

func defineGenerator(g *gorgonia.ExprGraph) *gan.GeneratorNet {

	/*
		input(250,250) => filters=12,size=3x3,conv(248,248) => filters=12,size=2x2,maxpool(124,124)
					   => filters=8,size=3x3,conv(122,122) => filters=8,size=2x2,maxpool(61,61)
					   => filters=5,size=3x3,conv(59,59) => filters=5,size=2x2,maxpool(29,29)
					   => filters=3,size=3x3,conv(27,27) => filters=3,size=2x2,maxpool(13,13)
					   => 3*flatten(13*13) => 25*linear(1, 13*13*3) => linear(225,25) => reshape(1,1,15,15)
	*/

	gen_shp0 := tensor.Shape{12, imgChannels, 3, 3}
	gen_w0 := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(gen_shp0...), gorgonia.WithName("generator_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp1 := tensor.Shape{8, 12 * imgChannels, 3, 3}
	gen_w1 := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(gen_shp1...), gorgonia.WithName("generator_w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp2 := tensor.Shape{5, 8 * imgChannels, 3, 3}
	gen_w2 := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(gen_shp2...), gorgonia.WithName("generator_w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp3 := tensor.Shape{3, 5 * imgChannels, 3, 3}
	gen_w3 := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(gen_shp3...), gorgonia.WithName("generator_w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp4 := tensor.Shape{25, 3 * 13 * 13}
	gen_w4 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(gen_shp4...), gorgonia.WithName("generator_w5"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp5 := tensor.Shape{imgHeight * imgWidth, 25}
	gen_w5 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(gen_shp5...), gorgonia.WithName("generator_w6"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	generator := gan.Generator(
		[]*gan.Layer{
			{
				WeightNode:   gen_w0,
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
				Probability: 0.6,
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
				WeightNode:   gen_w1,
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
				Probability: 0.5,
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
				WeightNode:   gen_w2,
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
				Probability: 0.2,
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
				WeightNode:   gen_w3,
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
				Probability: 0.2,
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
				WeightNode: gen_w4,
				BiasNode:   nil,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
			{
				WeightNode: gen_w5,
				BiasNode:   nil,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
			{
				Type:        gan.LayerReshape,
				Activation:  gan.NoActivation,
				ReshapeDims: []int{1, 1, imgHeight, imgWidth},
			},
		}...,
	)
	return generator
}
