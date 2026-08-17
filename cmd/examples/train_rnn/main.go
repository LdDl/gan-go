package main

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"

	gan "github.com/LdDl/gan-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	corpus = []string{
		"the quick brown fox jumps over the lazy dog",
		"the little cat sleeps under the warm sun",
		"a small bird sings in the green garden",
		"the old dog walks slowly through the park",
		"a young fox hides behind the tall tree",
	}
	sequenceLength = 4
	embeddingSize  = 16
	hiddenSize     = 32
	learningRate   = 0.01
	numOfEpochs    = 201
	evalPrint      = 40
)

type window struct {
	Input  []int
	Target []float64
}

func main() {
	rand.Seed(1337)

	/* Prepare dataset */
	vocab, wordToIndex := buildVocabulary(corpus)
	vocabSize := len(vocab)
	windows := buildWindows(corpus, wordToIndex, sequenceLength, vocabSize)
	fmt.Printf("Vocabulary size: %d\n", vocabSize)
	fmt.Printf("Number of training windows: %d\n", len(windows))

	/* Define Gorgonia's graph */
	netGraph := gorgonia.NewGraph()

	/* Define neural network */
	rnnNet := defineNet(netGraph, vocabSize)

	/* Prepare tensor for input values */
	inputNet := gorgonia.NewTensor(netGraph, gorgonia.Int, 1, gorgonia.WithShape(sequenceLength), gorgonia.WithName("rnn_train_input"))
	err := rnnNet.Fwd(inputNet)
	if err != nil {
		panic(err)
	}

	/* Prepare tensor for target values */
	targetNet := gorgonia.NewTensor(netGraph, gorgonia.Float64, 2, gorgonia.WithShape(sequenceLength, vocabSize), gorgonia.WithName("rnn_train_target"))

	/* Prepare variable for storing neural network's output */
	var netOut gorgonia.Value
	gorgonia.Read(rnnNet.Out(), &netOut)

	/* Prepare cost node */
	cost, err := gan.CrossEntropyLoss(rnnNet.Out(), targetNet)
	if err != nil {
		panic(err)
	}
	gorgonia.WithName("rnn_loss")(cost)

	/* Define gradients */
	_, err = gorgonia.Grad(cost, rnnNet.Learnables()...)
	if err != nil {
		panic(err)
	}

	/* Prepare variable for storing neural network's cost */
	var costOut gorgonia.Value
	gorgonia.Read(cost, &costOut)

	/* Define tape machine */
	tm := gorgonia.NewTapeMachine(netGraph, gorgonia.BindDualValues(rnnNet.Learnables()...))
	defer tm.Close()

	/* Initialize solver for evaluation graph */
	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(1), gorgonia.WithLearnRate(learningRate))

	/* Training process */
	for e := 0; e < numOfEpochs; e++ {
		for i := range windows {
			in := tensor.New(tensor.WithShape(sequenceLength), tensor.WithBacking(windows[i].Input))
			err = gorgonia.Let(inputNet, in)
			if err != nil {
				panic(err)
			}
			desired := tensor.New(tensor.WithShape(sequenceLength, vocabSize), tensor.WithBacking(windows[i].Target))
			err = gorgonia.Let(targetNet, desired)
			if err != nil {
				panic(err)
			}
			/* Run training step */
			err = tm.RunAll()
			if err != nil {
				panic(err)
			}
			err = solver.Step(gorgonia.NodesToValueGrads(rnnNet.Learnables()))
			if err != nil {
				panic(err)
			}
			tm.Reset()
		}
		// Shuffle training windows
		rand.Shuffle(len(windows), func(i, j int) { windows[i], windows[j] = windows[j], windows[i] })
		if e%evalPrint == 0 {
			fmt.Printf("Epoch %d:\n", e)
			fmt.Printf("\tLoss: %v\n", costOut)
		}
	}

	/* Test: continue every sentence of the corpus from its first words */
	fmt.Println("Start testing generator after final epoch")
	dummyTarget := tensor.New(tensor.WithShape(sequenceLength, vocabSize), tensor.WithBacking(make([]float64, sequenceLength*vocabSize)))
	for _, sentence := range corpus {
		words := strings.Fields(sentence)
		seed := make([]int, sequenceLength)
		for i := 0; i < sequenceLength; i++ {
			seed[i] = wordToIndex[words[i]]
		}
		generated := make([]string, 0, len(words))
		generated = append(generated, words[:sequenceLength]...)
		for len(generated) < len(words) {
			in := tensor.New(tensor.WithShape(sequenceLength), tensor.WithBacking(append([]int{}, seed...)))
			err = gorgonia.Let(inputNet, in)
			if err != nil {
				panic(err)
			}
			err = gorgonia.Let(targetNet, dummyTarget)
			if err != nil {
				panic(err)
			}
			err = tm.RunAll()
			if err != nil {
				panic(err)
			}
			probabilities := netOut.Data().([]float64)
			tm.Reset()
			// Next word is the most probable prediction at the last position of the window
			nextWord := argmax(probabilities[(sequenceLength-1)*vocabSize : sequenceLength*vocabSize])
			generated = append(generated, vocab[nextWord])
			seed = append(seed[1:], nextWord)
		}
		matched := "OK"
		if strings.Join(generated, " ") != sentence {
			matched = "MISMATCH"
		}
		fmt.Printf("\tSeed: %v\n", strings.Join(words[:sequenceLength], " "))
		fmt.Printf("\tContinued: %v [%s]\n", strings.Join(generated, " "), matched)
	}
}

func defineNet(g *gorgonia.ExprGraph, vocabSize int) *gan.DiscriminatorNet {
	embedding_w0 := gorgonia.NewTensor(g, gorgonia.Float64, 2, gorgonia.WithShape(vocabSize, embeddingSize), gorgonia.WithName("rnn_train_embedding_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	rnn_input_w0 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(embeddingSize, hiddenSize), gorgonia.WithName("rnn_train_input_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	rnn_hidden_w0 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, hiddenSize), gorgonia.WithName("rnn_train_hidden_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	rnn_b0 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, hiddenSize), gorgonia.WithName("rnn_train_b0"), gorgonia.WithInit(gorgonia.Zeroes()))

	linear_w0 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(vocabSize, hiddenSize), gorgonia.WithName("rnn_train_linear_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	net := gan.Discriminator(
		&gan.EmbeddingLayer{
			WeightNode:    embedding_w0,
			EmbeddingSize: embeddingSize,
		},
		&gan.RNNLayer{
			InputWeightNode:  rnn_input_w0,
			HiddenWeightNode: rnn_hidden_w0,
			BiasNode:         rnn_b0,
			HiddenSize:       hiddenSize,
		},
		&gan.LinearLayer{
			WeightNode: linear_w0,
			Activation: gan.WithActivationOptions(gan.Softmax, gan.Options{Axis: []int{1}}),
		},
	)
	return net
}

// buildVocabulary Collects sorted unique words of the corpus
func buildVocabulary(sentences []string) ([]string, map[string]int) {
	unique := make(map[string]bool)
	for _, sentence := range sentences {
		for _, word := range strings.Fields(sentence) {
			unique[word] = true
		}
	}
	vocab := make([]string, 0, len(unique))
	for word := range unique {
		vocab = append(vocab, word)
	}
	sort.Strings(vocab)
	wordToIndex := make(map[string]int, len(vocab))
	for i, word := range vocab {
		wordToIndex[word] = i
	}
	return vocab, wordToIndex
}

// buildWindows Prepares sliding windows: for tokens t[i]...t[i+n-1] target is one-hot encoded t[i+1]...t[i+n]
func buildWindows(sentences []string, wordToIndex map[string]int, n, vocabSize int) []window {
	windows := []window{}
	for _, sentence := range sentences {
		words := strings.Fields(sentence)
		tokens := make([]int, len(words))
		for i, word := range words {
			tokens[i] = wordToIndex[word]
		}
		for i := 0; i+n < len(tokens); i++ {
			target := make([]float64, n*vocabSize)
			for j := 0; j < n; j++ {
				target[j*vocabSize+tokens[i+1+j]] = 1.0
			}
			windows = append(windows, window{
				Input:  tokens[i : i+n],
				Target: target,
			})
		}
	}
	return windows
}

func argmax(values []float64) int {
	best := 0
	for i := range values {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}
