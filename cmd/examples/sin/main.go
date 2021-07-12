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

func generateX() float64 {
	return 2 * math.Pi * rand.Float64()
}

func generateY(x float64) float64 {
	return math.Sin(x)
}

var (
	outputFolder    = "./output"
	batchSize       = 16
	latentSpaceSize = 2
	numEpoches      = 400
	numTestSamples  = 300
	evalPrint       = 20
)

func main() {
	// Initialize seed with constant value to reproduce results
	rand.Seed(1337)

	// Prepare synthetic data
	trainDataLength := 1024
	trainSet, err := gan.GenerateTrainingSet(trainDataLength, generateX, generateY)
	if err != nil {
		panic(err)
	}

	// Extract X and Y(X) values for charts plotting
	slicedXAxis, err := trainSet.TrainData.Slice(nil, gorgonia.S(0))
	if err != nil {
		panic(err)
	}
	slicedYAxis, err := trainSet.TrainData.Slice(nil, gorgonia.S(1))
	if err != nil {
		panic(err)
	}

	// Plot reference function
	err = gan.PlotXY(slicedXAxis.Materialize(), slicedYAxis.Materialize(), fmt.Sprintf("%s/reference_function.png", outputFolder))
	if err != nil {
		panic(err)
	}

	// Define graph for GAN feedforward and Generator training
	ganGraph := gorgonia.NewGraph()
	// Define graph for Discriminator training
	trainDiscriminatorGraph := gorgonia.NewGraph()

	// Define Generator on GAN's evaluation graph
	definedGenerator := defineGenerator(ganGraph)
	// Initialize Generator feedforward
	inputGenerator := gorgonia.NewMatrix(ganGraph, gorgonia.Float64, gorgonia.WithShape(batchSize, latentSpaceSize), gorgonia.WithName("generator_input"))
	err = definedGenerator.Fwd(inputGenerator, batchSize)
	if err != nil {
		panic(err)
	}

	// Define Discriminator on its own evaluation graph
	discriminatorTrain := defineDiscriminator(trainDiscriminatorGraph)
	// Initialize Discriminator feedforward
	inputDiscriminatorTrain := gorgonia.NewMatrix(trainDiscriminatorGraph, gorgonia.Float64, gorgonia.WithShape(2*batchSize, 2), gorgonia.WithName("discriminator_train_input"))
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

	// Define loss function for GAN as
	// loss{i} = (gan_out{i} - target{i})^2
	targetDiscriminatorGAN := gorgonia.NewMatrix(ganGraph, gorgonia.Float64, gorgonia.WithShape(definedGAN.Out().Shape()...), gorgonia.WithName("gan_discriminator_target"))
	lossDiscriminatorGAN := gorgonia.Must(
		gorgonia.Square(
			gorgonia.Must(
				gorgonia.Sub(
					definedGAN.Out(),
					targetDiscriminatorGAN,
				),
			),
		),
	)
	gorgonia.WithName("gan_discriminator_loss")(lossDiscriminatorGAN)
	// Define cost function for GAN as
	// cost = AVG(loss{i=1...N})
	cost := gorgonia.Must(gorgonia.Mean(lossDiscriminatorGAN))
	// Define gradients for GAN
	_, err = gorgonia.Grad(cost, definedGAN.Learnables()...)
	if err != nil {
		panic(err)
	}

	// Define loss function for Discriminator in training mode as
	// loss{i} = (gan_out{i} - target{i})^2
	targetDiscriminatorTrain := gorgonia.NewMatrix(trainDiscriminatorGraph, gorgonia.Float64, gorgonia.WithShape(2*batchSize, 1), gorgonia.WithName("discriminator_target"))
	losses0_train := gorgonia.Must(
		gorgonia.Square(
			gorgonia.Must(
				gorgonia.Sub(
					discriminatorTrain.Out(),
					targetDiscriminatorTrain,
				),
			),
		),
	)
	gorgonia.WithName("discriminator_loss")(losses0_train)
	// Define cost function for Discriminator in training mode as
	// cost = AVG(loss{i=1...N})
	costDiscriminatorTrain := gorgonia.Must(gorgonia.Mean(losses0_train))
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
			xVal, err = trainSet.TrainData.Slice(gan.SlicerOneStep{StartIdx: start, EndIdx: end})
			if err != nil {
				panic(err)
			}
			// Crutch for reshaping vector to matrix (keep in mind batch_size)
			err = xVal.Reshape(batchSize, 2)
			if err != nil {
				panic(err)
			}

			real_samples_labels := tensor.Ones(tensor.Float64, batchSize, 1)
			latentSpaceSamples := gan.NormRandDense(batchSize, latentSpaceSize)
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

			latentSpaceSamplesGenerated := gan.NormRandDense(batchSize, latentSpaceSize)
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

				testSamplesTensor, err := gan.GenerateNormTestSamples(tmGenerator, tmDisTrain, inputGenerator, inputDiscriminatorTrain, generatedSamples, numTestSamples, batchSize, latentSpaceSize, nil)
				if err != nil {
					panic(err)
				}
				// Extract X and Y(X) values for charts plotting
				slicedXAxis, err := testSamplesTensor.Slice(nil, gorgonia.S(0))
				if err != nil {
					panic(err)
				}
				slicedYAxis, err := testSamplesTensor.Slice(nil, gorgonia.S(1))
				if err != nil {
					panic(err)
				}
				// Plot output of Generator of certain epoch
				err = gan.PlotXY(slicedXAxis.Materialize(), slicedYAxis.Materialize(), fmt.Sprintf("%s/gen_reference_func_%d.png", outputFolder, epoch))
				if err != nil {
					panic(err)
				}
			}
		}
	}

	// Final test of Generator
	fmt.Println("Start testing generator after final epoch")
	testSamplesTensor, err := gan.GenerateNormTestSamples(tmGenerator, tmDisTrain, inputGenerator, inputDiscriminatorTrain, generatedSamples, numTestSamples, batchSize, latentSpaceSize, nil)
	if err != nil {
		panic(err)
	}
	// Extract X and Y(X) values for charts plotting
	slicedXAxis, err = testSamplesTensor.Slice(nil, gorgonia.S(0))
	if err != nil {
		panic(err)
	}
	slicedYAxis, err = testSamplesTensor.Slice(nil, gorgonia.S(1))
	if err != nil {
		panic(err)
	}
	// Plot output of Generator of certain epoch
	err = gan.PlotXY(slicedXAxis.Materialize(), slicedYAxis.Materialize(), fmt.Sprintf("%s/gen_reference_func_final.png", outputFolder))
	if err != nil {
		panic(err)
	}
}

func defineDiscriminator(g *gorgonia.ExprGraph) *gan.Discriminator {
	dis_shp0 := tensor.Shape{256, 2}

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

	discriminator := gan.Discriminator{
		Layers: []*gan.Layer{
			{
				WeightNode: dis_w0,
				BiasNode:   dis_b0,
				Type:       gan.LayerLinear,
				Activation: gan.Rectify,
			},
			{
				WeightNode: dis_w1,
				BiasNode:   dis_b1,
				Type:       gan.LayerLinear,
				Activation: gan.Rectify,
			},
			{
				WeightNode: dis_w2,
				BiasNode:   dis_b2,
				Type:       gan.LayerLinear,
				Activation: gan.Rectify,
			},
			{
				WeightNode: dis_w3,
				BiasNode:   dis_b3,
				Type:       gan.LayerLinear,
				Activation: gan.Rectify,
			},
			{
				WeightNode: dis_w5,
				BiasNode:   dis_b5,
				Activation: gan.Sigmoid,
			},
		},
	}
	return &discriminator
}

func defineGenerator(g *gorgonia.ExprGraph) *gan.Generator {
	gen_shp0 := tensor.Shape{16, latentSpaceSize}
	gen_b0 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, gen_shp0[0]), gorgonia.WithName("generator_b0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	gen_w0 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(gen_shp0...), gorgonia.WithName("generator_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp1 := tensor.Shape{32, 16}
	gen_b1 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, gen_shp1[0]), gorgonia.WithName("generator_b1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	gen_w1 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(gen_shp1...), gorgonia.WithName("generator_w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp2 := tensor.Shape{16, 32}
	gen_b2 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, gen_shp2[0]), gorgonia.WithName("generator_b2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	gen_w2 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(gen_shp2...), gorgonia.WithName("generator_w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	gen_shp3 := tensor.Shape{2, 16}
	gen_b3 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, gen_shp3[0]), gorgonia.WithName("generator_b3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	gen_w3 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(gen_shp3...), gorgonia.WithName("generator_w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	generator := gan.Generator{
		Layers: []*gan.Layer{
			{
				WeightNode: gen_w0,
				BiasNode:   gen_b0,
				Type:       gan.LayerLinear,
				Activation: gan.Rectify,
			},
			{
				WeightNode: gen_w1,
				BiasNode:   gen_b1,
				Type:       gan.LayerLinear,
				Activation: gan.Rectify,
			},
			{
				WeightNode: gen_w2,
				BiasNode:   gen_b2,
				Type:       gan.LayerLinear,
				Activation: gan.Rectify,
			},
			{
				WeightNode: gen_w3,
				BiasNode:   gen_b3,
				Type:       gan.LayerLinear,
				Activation: gan.NoActivation,
			},
		},
	}

	return &generator
}
