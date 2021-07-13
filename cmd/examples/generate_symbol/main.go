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
	outputFolder    = "./output"
	dataFile        = "./training_data.csv"
	batchSize       = 1
	latentSpaceSize = 150
	symbolHeight    = 10
	symbolWidth     = 8
	numEpoches      = 210
	numTestSamples  = 1
	evalPrint       = 30
)

func genSyntheticData(numSamples int) *gan.TrainSet {
	// Generate 'H' char in binary representation
	f64data := []float64{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 1, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 1, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	}
	data := tensor.New(tensor.WithShape(1, symbolHeight*symbolWidth), tensor.WithBacking(f64data))

	fmt.Println("Reference data:")
	for x := 0; x < symbolHeight; x++ {
		fmt.Printf("\t")
		for y := 0; y < symbolWidth; y++ {
			r := math.Round(f64data[x*symbolWidth+y])
			if r == -0 {
				r = 0
			}
			fmt.Printf("%.0f ", r)
		}
		fmt.Println()
	}

	labels := tensor.Ones(tensor.Float64, numSamples, 1)
	return &gan.TrainSet{
		TrainData:  data,
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
	inputGenerator := gorgonia.NewMatrix(ganGraph, gorgonia.Float64, gorgonia.WithShape(batchSize, latentSpaceSize), gorgonia.WithName("generator_input"))
	err := definedGenerator.Fwd(inputGenerator, batchSize)
	if err != nil {
		panic(err)
	}

	// Define Discriminator on its own evaluation graph
	discriminatorTrain := defineDiscriminator(trainDiscriminatorGraph)
	// Initialize Discriminator feedforward
	inputDiscriminatorTrain := gorgonia.NewMatrix(trainDiscriminatorGraph, gorgonia.Float64, gorgonia.WithShape(2*batchSize, symbolHeight*symbolWidth), gorgonia.WithName("discriminator_train_input"))
	discriminatorTrain.Fwd(inputDiscriminatorTrain, 2*batchSize)

	// Define GAN on the same evaluation graph as Generator has been defined
	definedGAN, err := gan.NewGAN(ganGraph, definedGenerator, discriminatorTrain)
	if err != nil {
		panic(err)
	}
	definedGAN.Fwd(batchSize)

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

	targetDiscriminatorGAN := gorgonia.NewMatrix(ganGraph, gorgonia.Float64, gorgonia.WithShape(definedGAN.Out().Shape()...), gorgonia.WithName("gan_discriminator_target"))
	/* Define cost for GAN as*/
	cost, err := gan.MSELoss(definedGAN.Out(), targetDiscriminatorGAN)
	if err != nil {
		panic(err)
	}
	gorgonia.WithName("gan_discriminator_loss")(cost)
	// Define gradients for GAN
	_, err = gorgonia.Grad(cost, definedGAN.Learnables()...)
	if err != nil {
		panic(err)
	}

	targetDiscriminatorTrain := gorgonia.NewMatrix(trainDiscriminatorGraph, gorgonia.Float64, gorgonia.WithShape(2*batchSize, 1), gorgonia.WithName("discriminator_target"))
	/* Define cost for Distriminator in training mode as*/
	costDiscriminatorTrain, err := gan.MSELoss(discriminatorTrain.Out(), targetDiscriminatorTrain)
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
	solverGAN := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(0.001))
	// Tape machine for Discriminator [in training mode] evaluation graph
	tmDisTrain := gorgonia.NewTapeMachine(trainDiscriminatorGraph, gorgonia.BindDualValues(discriminatorTrain.Learnables()...))
	defer tmDisTrain.Close()
	// Solver for Discriminator [in training mode] evaluation graph
	solverDiscriminatorTrain := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(0.001))

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
			latentSpaceSamples := gan.UniformRandDense(batchSize, latentSpaceSize)
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

			latentSpaceSamplesGenerated := gan.UniformRandDense(batchSize, latentSpaceSize)
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
				testSamplesTensor, err := gan.GenerateUniformTestSamples(tmGenerator, tmDisTrain, inputGenerator, inputDiscriminatorTrain, generatedSamples, numTestSamples, batchSize, latentSpaceSize, nil)
				if err != nil {
					panic(err)
				}
				testData := testSamplesTensor.Materialize().Data().([]float64)
				for x := 0; x < symbolHeight; x++ {
					fmt.Printf("\t")
					for y := 0; y < symbolWidth; y++ {
						r := math.Round(testData[x*symbolWidth+y])
						if r == -0 {
							r = 0
						}
						fmt.Printf("%.0f ", r)
					}
					fmt.Println()
				}
			}
		}
	}

	// Final test of Generator
	fmt.Println("Start testing generator after final epoch")
	testSamplesTensor, err := gan.GenerateUniformTestSamples(tmGenerator, tmDisTrain, inputGenerator, inputDiscriminatorTrain, generatedSamples, numTestSamples, batchSize, latentSpaceSize, nil)
	if err != nil {
		panic(err)
	}
	testData := testSamplesTensor.Materialize().Data().([]float64)
	for x := 0; x < symbolHeight; x++ {
		fmt.Printf("\t")
		for y := 0; y < symbolWidth; y++ {
			r := math.Round(testData[x*symbolWidth+y])
			if r == -0 {
				r = 0
			}
			fmt.Printf("%.0f ", r)
		}
		fmt.Println()
	}
}

func defineDiscriminator(g *gorgonia.ExprGraph) *gan.DiscriminatorNet {
	dis_shp0 := tensor.Shape{256, symbolHeight * symbolWidth}

	dis_b0 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, dis_shp0[0]), gorgonia.WithName("discriminator_train_b0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	dis_w0 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(dis_shp0...), gorgonia.WithName("discriminator_train_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	dis_shp1 := tensor.Shape{128, 256}
	dis_b1 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, dis_shp1[0]), gorgonia.WithName("discriminator_train_b1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	dis_w1 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(dis_shp1...), gorgonia.WithName("discriminator_train_w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	dis_shp2 := tensor.Shape{64, 128}
	dis_b2 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, dis_shp2[0]), gorgonia.WithName("discriminator_train_b2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	dis_w2 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(dis_shp2...), gorgonia.WithName("discriminator_train_w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	dis_shp3 := tensor.Shape{32, 64}
	dis_b3 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, dis_shp3[0]), gorgonia.WithName("discriminator_train_b3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	dis_w3 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(dis_shp3...), gorgonia.WithName("discriminator_train_w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	dis_shp5 := tensor.Shape{1, 32}
	dis_b5 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, dis_shp5[0]), gorgonia.WithName("discriminator_train_b5"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	dis_w5 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(dis_shp5...), gorgonia.WithName("discriminator_train_w5"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	discriminator := gan.Discriminator(
		[]*gan.Layer{
			{
				WeightNode: dis_w0,
				BiasNode:   dis_b0,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
			{
				WeightNode: dis_w1,
				BiasNode:   dis_b1,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
			{
				WeightNode: dis_w2,
				BiasNode:   dis_b2,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
			{
				WeightNode: dis_w3,
				BiasNode:   dis_b3,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
			{
				WeightNode: dis_w5,
				BiasNode:   dis_b5,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
		}...,
	)
	return discriminator
}

func defineGenerator(g *gorgonia.ExprGraph) *gan.GeneratorNet {
	gen_shp0 := tensor.Shape{16, latentSpaceSize}
	gen_b0 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, gen_shp0[0]), gorgonia.WithName("generator_b0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	gen_w0 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(gen_shp0...), gorgonia.WithName("generator_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp1 := tensor.Shape{32, 16}
	gen_b1 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, gen_shp1[0]), gorgonia.WithName("generator_b1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	gen_w1 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(gen_shp1...), gorgonia.WithName("generator_w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp2 := tensor.Shape{16, 32}
	gen_b2 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, gen_shp2[0]), gorgonia.WithName("generator_b2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	gen_w2 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(gen_shp2...), gorgonia.WithName("generator_w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp3 := tensor.Shape{symbolHeight * symbolWidth, 16}
	gen_b3 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, gen_shp3[0]), gorgonia.WithName("generator_b3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	gen_w3 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(gen_shp3...), gorgonia.WithName("generator_w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	generator := gan.Generator(
		[]*gan.Layer{
			{
				WeightNode: gen_w0,
				BiasNode:   gen_b0,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
			{
				WeightNode: gen_w1,
				BiasNode:   gen_b1,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
			{
				WeightNode: gen_w2,
				BiasNode:   gen_b2,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
			{
				WeightNode: gen_w3,
				BiasNode:   gen_b3,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
		}...,
	)

	return generator
}
