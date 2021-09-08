package main

import (
	"fmt"
	"math"
	"math/rand"

	gan "github.com/LdDl/gan-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const (
	BadPerfomance     = 0.0
	AveragePerfomance = 0.5
	GoodPerfomance    = 1.0
)

type Evaluation struct {
	Value  string
	Hashed []int
	Label  []float64
}

func main() {
	rand.Seed(1337)

	vocabulary := 50
	evaluations := []Evaluation{
		{
			Value:  "Well done!",
			Hashed: []int{},
			Label:  []float64{GoodPerfomance},
		},
		{
			Value:  "Good work",
			Hashed: []int{},
			Label:  []float64{GoodPerfomance},
		},
		{
			Value:  "Great effort",
			Hashed: []int{},
			Label:  []float64{GoodPerfomance},
		},
		{
			Value:  "nice work",
			Hashed: []int{},
			Label:  []float64{GoodPerfomance},
		},
		{
			Value:  "Excellent!",
			Hashed: []int{},
			Label:  []float64{GoodPerfomance},
		},
		{
			Value:  "Weak",
			Hashed: []int{},
			Label:  []float64{BadPerfomance},
		},
		{
			Value:  "Poor effort!",
			Hashed: []int{},
			Label:  []float64{BadPerfomance},
		},
		{
			Value:  "not good",
			Hashed: []int{},
			Label:  []float64{BadPerfomance},
		},
		{
			Value:  "poor work",
			Hashed: []int{},
			Label:  []float64{BadPerfomance},
		},
		{
			Value:  "Could be way better.",
			Hashed: []int{},
			Label:  []float64{BadPerfomance},
		},
		{
			Value:  "average",
			Hashed: []int{},
			Label:  []float64{AveragePerfomance},
		},
		{
			Value:  "middle level",
			Hashed: []int{},
			Label:  []float64{AveragePerfomance},
		},
		{
			Value:  "ordinary stuff",
			Hashed: []int{},
			Label:  []float64{AveragePerfomance},
		},
		{
			Value:  "boilerplate",
			Hashed: []int{},
			Label:  []float64{AveragePerfomance},
		},
		{
			Value:  "standart approach",
			Hashed: []int{},
			Label:  []float64{AveragePerfomance},
		},
	}

	maxPadding := 5
	for i := range evaluations {
		tmp, err := gan.HashingTrick(evaluations[i].Value, vocabulary, gan.HASH_SHA256)
		if err != nil {
			fmt.Printf("Can't init one-hot data for string '%s' due the error: %s\n", evaluations[i].Value, err.Error())
			return
		}
		evaluations[i].Hashed = int64ToInt(gan.PaddingInt64Slice(tmp, maxPadding, gan.PADDING_POST))
	}

	/* Define Gorgonia's graph */
	netGraph := gorgonia.NewGraph()

	/* Define structure of neural network */
	embeddingDim := 12
	simpleNet := defineNet(netGraph, vocabulary, maxPadding, embeddingDim)

	/* Prepare tensor for input values */
	inputShape := tensor.Shape{maxPadding}
	batchSize := 1
	inputNet := gorgonia.NewTensor(netGraph, gorgonia.Int, 1, gorgonia.WithShape(inputShape...), gorgonia.WithName("discriminator_train_input"))
	err := simpleNet.Fwd(inputNet, batchSize)
	if err != nil {
		panic(err)
	}

	/* Prepare tensor for label values */
	targetNet := gorgonia.NewTensor(netGraph, gorgonia.Float64, 2, gorgonia.WithShape(1, 1), gorgonia.WithName("discriminator_label"))
	/* Prepare variable for storing neural network's output */
	var netOut gorgonia.Value
	gorgonia.Read(simpleNet.Out(), &netOut)
	/* Prepare cost node */
	cost, err := gan.MSELoss(simpleNet.Out(), targetNet)
	if err != nil {
		panic(err)
	}
	gorgonia.WithName("discriminator_loss")(cost)

	/* Define gradients */
	_, err = gorgonia.Grad(cost, simpleNet.Learnables()...)
	if err != nil {
		panic(err)
	}

	/* Prepare variable for storing neural network's cost */
	var costOut gorgonia.Value
	gorgonia.Read(cost, &costOut)

	/* Define tape machine */
	tm := gorgonia.NewTapeMachine(netGraph, gorgonia.BindDualValues(simpleNet.Learnables()...))
	defer tm.Close()

	/* Initialize solver for evaluation graph */
	learning_rate := 0.01
	numOfEpochs := 200
	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate))

	// Run through all epochs
	evalPrint := 40
	for e := 0; e < numOfEpochs; e++ {
		// Run through each hardcoded sample
		for i := 0; i < len(evaluations); i++ {
			// Prepare input
			in := tensor.New(tensor.WithShape(inputShape...), tensor.WithBacking(evaluations[i].Hashed))
			err = gorgonia.Let(inputNet, in)
			if err != nil {
				panic(err)
			}
			// Prepare training label
			desired := tensor.New(tensor.WithShape(1, 1), tensor.WithBacking(evaluations[i].Label))
			err = gorgonia.Let(targetNet, desired)
			if err != nil {
				panic(err)
			}
			/* Run training step */
			err = tm.RunAll()
			if err != nil {
				panic(err)
			}
			err = solver.Step(gorgonia.NodesToValueGrads(simpleNet.Learnables()))
			if err != nil {
				panic(err)
			}
			tm.Reset()
		}
		// Suffle training slice
		rand.Shuffle(len(evaluations), func(i, j int) { evaluations[i], evaluations[j] = evaluations[j], evaluations[i] })
		if e%evalPrint == 0 {
			fmt.Printf("Epoch %d:\n", e)
			fmt.Printf("\tDiscriminator's loss: %v\n", costOut)
		}
	}

	/* Test */
	for i := 0; i < len(evaluations); i++ {
		// Prepare input
		in := tensor.New(tensor.WithShape(inputShape...), tensor.WithBacking(evaluations[i].Hashed))
		err = gorgonia.Let(inputNet, in)
		if err != nil {
			panic(err)
		}

		// Prepare training label
		desired := tensor.New(tensor.WithShape(1, 1), tensor.WithBacking(evaluations[i].Label))
		err = gorgonia.Let(targetNet, desired)
		if err != nil {
			panic(err)
		}

		/* Run training step */
		err = tm.RunAll()
		if err != nil {
			panic(err)
		}
		fmt.Printf("Text assessment: %s\n", evaluations[i].Value)
		fmt.Printf("\tIts hashed value: %d\n", evaluations[i].Hashed)
		fmt.Printf("\tIts defined numerical assessment: %.1f\n", evaluations[i].Label[0])
		fmt.Printf("\tIts evaluated numerical assessment: %.1f\n", netOut.Data().([]float64)[0])
		fmt.Printf("\tDifference between defined and evaluated: %.1f\n", math.Abs(evaluations[i].Label[0]-netOut.Data().([]float64)[0]))
		tm.Reset()
	}
}

func defineNet(g *gorgonia.ExprGraph, vocabulary, maxWords, embeddingDim int) *gan.DiscriminatorNet {
	dis_shp0 := tensor.Shape{vocabulary, embeddingDim}
	dis_w0 := gorgonia.NewTensor(g, gorgonia.Float64, 2, gorgonia.WithShape(dis_shp0...), gorgonia.WithName("discriminator_train_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	dis_shp1 := tensor.Shape{1, maxWords * embeddingDim}
	dis_w1 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(dis_shp1...), gorgonia.WithName("discriminator_train_w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	discriminator := gan.Discriminator(
		[]*gan.Layer{
			{
				WeightNode: dis_w0,
				BiasNode:   nil,
				Type:       gan.LayerEmbedding,
				Options: &gan.Options{
					EmbeddingSize: embeddingDim,
				},
			},
			{
				Type: gan.LayerFlatten,
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

func int64ToInt(s []int64) []int {
	ans := make([]int, len(s))
	for i := range s {
		ans[i] = int(s[i])
	}
	return ans
}
