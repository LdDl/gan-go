package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"

	gan "github.com/LdDl/gan-go"
	"github.com/pkg/errors"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	rand.Seed(1337)

	/* Prepare dataset */
	batchSize := 256
	sequenceLength := 4
	datasetFileName := "jokes.csv"
	dataset, err := initDataset(datasetFileName, batchSize, sequenceLength)
	if err != nil {
		fmt.Println(err)
		return
	}

	/* Define Gorgonia's graph */
	netGraph := gorgonia.NewGraph()

	/* Define LSTM neural network*/
	inputDimension := 128
	hiddenSize := 128
	lstmModel := defineNet(netGraph, inputDimension, hiddenSize, len(dataset.uniqueWords))
	_ = lstmModel

}

func defineNet(g *gorgonia.ExprGraph, lstmDims, lstmHidden, vocabSize int) *gan.DiscriminatorNet {
	dis_shp0 := tensor.Shape{vocabSize, lstmHidden}
	dis_w0 := gorgonia.NewTensor(g, gorgonia.Float64, 2, gorgonia.WithShape(dis_shp0...), gorgonia.WithName("discriminator_train_w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	dis_input_shp0 := tensor.Shape{1, lstmDims, lstmHidden * 4}
	dis_input_w0 := gorgonia.NewTensor(g, gorgonia.Float64, 3, gorgonia.WithShape(dis_input_shp0...), gorgonia.WithName("discriminator_lstm_input_w0"), gorgonia.WithInit(gorgonia.Ones()))
	dis_hidden_shp0 := tensor.Shape{1, lstmHidden, lstmHidden * 4}
	dis_hidden_w0 := gorgonia.NewTensor(g, gorgonia.Float64, 3, gorgonia.WithShape(dis_hidden_shp0...), gorgonia.WithName("discriminator_lstm_hidden_w0"), gorgonia.WithInit(gorgonia.Ones()))
	dis_dummy_hidden_w0 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, lstmHidden), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("discriminator_lstm_dimmy_hidden_0"))
	dis_dummy_cell_w0 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, lstmHidden), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("discriminator_lstm_dummy_cell_0"))

	dis_input_shp1 := tensor.Shape{1, lstmDims, lstmHidden * 4}
	dis_input_w1 := gorgonia.NewTensor(g, gorgonia.Float64, 3, gorgonia.WithShape(dis_input_shp1...), gorgonia.WithName("discriminator_lstm_input_w1"), gorgonia.WithInit(gorgonia.Ones()))
	dis_hidden_shp1 := tensor.Shape{1, lstmHidden, lstmHidden * 4}
	dis_hidden_w1 := gorgonia.NewTensor(g, gorgonia.Float64, 3, gorgonia.WithShape(dis_hidden_shp1...), gorgonia.WithName("discriminator_lstm_hidden_w1"), gorgonia.WithInit(gorgonia.Ones()))
	dis_dummy_hidden_w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, lstmHidden), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("discriminator_lstm_dimmy_hidden_1"))
	dis_dummy_cell_w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, lstmHidden), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("discriminator_lstm_dummy_cell_1"))

	dis_input_shp2 := tensor.Shape{1, lstmDims, lstmHidden * 4}
	dis_input_w2 := gorgonia.NewTensor(g, gorgonia.Float64, 3, gorgonia.WithShape(dis_input_shp2...), gorgonia.WithName("discriminator_lstm_input_w2"), gorgonia.WithInit(gorgonia.Ones()))
	dis_hidden_shp2 := tensor.Shape{1, lstmHidden, lstmHidden * 4}
	dis_hidden_w2 := gorgonia.NewTensor(g, gorgonia.Float64, 3, gorgonia.WithShape(dis_hidden_shp2...), gorgonia.WithName("discriminator_lstm_hidden_w2"), gorgonia.WithInit(gorgonia.Ones()))
	dis_dummy_hidden_w2 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, lstmHidden), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("discriminator_lstm_dimmy_hidden_2"))
	dis_dummy_cell_w2 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, lstmHidden), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("discriminator_lstm_dummy_cell_2"))

	dis_shp3 := tensor.Shape{1, vocabSize * lstmHidden}
	dis_w3 := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(dis_shp3...), gorgonia.WithName("discriminator_train_w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	discriminator := gan.Discriminator(
		[]*gan.Layer{
			{
				WeightNode: dis_w0,
				BiasNode:   nil,
				Type:       gan.LayerEmbedding,
				Options: &gan.Options{
					EmbeddingSize: lstmHidden,
				},
			},
			{
				WeightNode: dis_input_w0,
				Type:       gan.LayerLSTM,
				Options: &gan.Options{
					LSTM: &gan.LSTMOptions{
						InputDimension:      lstmDims,
						HiddenSize:          lstmHidden,
						Activation:          gan.Tanh,
						ReccurentActivation: gan.Sigmoid,
						HiddenNode:          dis_hidden_w0,
						DummyHidden:         dis_dummy_hidden_w0,
						DummyCell:           dis_dummy_cell_w0,
					},
				},
			},
			{
				WeightNode: dis_input_w1,
				Type:       gan.LayerLSTM,
				Options: &gan.Options{
					LSTM: &gan.LSTMOptions{
						InputDimension:      lstmDims,
						HiddenSize:          lstmHidden,
						Activation:          gan.Tanh,
						ReccurentActivation: gan.Sigmoid,
						HiddenNode:          dis_hidden_w1,
						DummyHidden:         dis_dummy_hidden_w1,
						DummyCell:           dis_dummy_cell_w1,
					},
				},
			},
			{
				WeightNode: dis_input_w2,
				Type:       gan.LayerLSTM,
				Options: &gan.Options{
					LSTM: &gan.LSTMOptions{
						InputDimension:      lstmDims,
						HiddenSize:          lstmHidden,
						Activation:          gan.Tanh,
						ReccurentActivation: gan.Sigmoid,
						HiddenNode:          dis_hidden_w2,
						DummyHidden:         dis_dummy_hidden_w2,
						DummyCell:           dis_dummy_cell_w2,
					},
				},
			},
			{
				Type: gan.LayerFlatten,
			},
			{
				WeightNode: dis_w3,
				BiasNode:   nil,
				Type:       gan.LayerLinear,
				Activation: gan.Sigmoid,
			},
		}...,
	)
	return discriminator
}

type Dataset struct {
	allWords       []string
	uniqueWords    []string
	indicesToWords map[int]string
	wordsToIndices map[string]int
	wordsIndices   []int

	batchSize      int
	sequenceLength int

	batchedX [][][]int
	batchedY [][][]int
}

func initDataset(datasetFileName string, batchSize, sequenceLength int) (*Dataset, error) {
	dataset, err := readCSV("jokes.csv", ',', true)
	if err != nil {
		return nil, errors.Wrap(err, "Can't read CSV")
	}
	ret := &Dataset{
		allWords:       extractWords(dataset),
		batchSize:      batchSize,
		sequenceLength: sequenceLength,
	}
	fmt.Println("Number of words in dataset:", len(ret.allWords))
	ret.uniqueWords = extractUniqueWords(ret.allWords)
	fmt.Println("Number of unique words in dataset:", len(ret.uniqueWords))
	ret.indicesToWords = convertIndicesToWords(ret.uniqueWords)
	ret.wordsToIndices = convertWordsToIndices(ret.uniqueWords)
	ret.wordsIndices = extractWordsIndices(ret.allWords, ret.wordsToIndices)
	ret.prepareForTrain()
	return ret, nil
}

func (dataset *Dataset) prepareForTrain() {
	dataLen := len(dataset.wordsIndices)
	// [batches][sequences][words]int
	dataset.batchedX = make([][][]int, (dataLen+dataset.batchSize-1)/dataset.batchSize)
	dataset.batchedY = make([][][]int, (dataLen+dataset.batchSize-1)/dataset.batchSize)
	prev := 0
	i := 0
	till := dataLen - dataset.batchSize
	for prev < till {
		next := prev + dataset.batchSize
		dataset.batchedX[i] = [][]int{}
		dataset.batchedY[i] = [][]int{}
		for idx := prev; idx < next; idx++ {
			xvalue, yvalues := dataset.getXYByIndex(idx)
			dataset.batchedX[i] = append(dataset.batchedX[i], xvalue)
			dataset.batchedY[i] = append(dataset.batchedY[i], yvalues)
		}
		prev = next
		i++
	}
	for idx := prev; idx < dataLen-dataset.sequenceLength; idx++ {
		xvalue, yvalues := dataset.getXYByIndex(idx)
		dataset.batchedX[i] = append(dataset.batchedX[i], xvalue)
		dataset.batchedY[i] = append(dataset.batchedY[i], yvalues)
	}
}

func (dataset *Dataset) getXYByIndex(idx int) ([]int, []int) {
	return getWordByIndex(dataset.wordsIndices, idx, dataset.sequenceLength)
}

func getWordByIndex(wordsIndices []int, idx, sequenceIdx int) ([]int, []int) {
	x := wordsIndices[idx : idx+sequenceIdx]
	y := wordsIndices[idx+1 : idx+sequenceIdx+1]
	return x, y
}

func extractWordsIndices(allWords []string, wordsToIndices map[string]int) []int {
	ret := make([]int, len(allWords))
	for i, word := range allWords {
		ret[i] = wordsToIndices[word]
	}
	return ret
}

func convertIndicesToWords(uniqueWords []string) map[int]string {
	idxToWord := make(map[int]string, len(uniqueWords))
	for i, word := range uniqueWords {
		idxToWord[i] = word
	}
	return idxToWord
}

func convertWordsToIndices(uniqueWords []string) map[string]int {
	wrdToIndex := make(map[string]int, len(uniqueWords))
	for i, word := range uniqueWords {
		wrdToIndex[word] = i
	}
	return wrdToIndex
}

func extractWords(sentences [][]string) []string {
	words := []string{}
	for _, sentence := range sentences {
		if len(sentences) < 2 {
			continue
		}
		wordsLocal := strings.Split(sentence[1], " ")
		words = append(words, wordsLocal...)
	}
	return words
}

type pair struct {
	word      string
	frequence int
}

func extractUniqueWords(words []string) []string {
	un := make(map[string]int)
	for _, word := range words {
		if _, ok := un[word]; !ok {
			un[word] = 0
		}
		un[word]++
	}
	// Before return unique words let's sort them in ascending order by frequency
	wc := make([]pair, 0, len(un))
	for w, c := range un {
		wc = append(wc, pair{w, c})
	}
	sort.Slice(wc, func(i, j int) bool {
		return wc[i].frequence > wc[j].frequence
	})
	ret := make([]string, len(wc))
	for i := range wc {
		ret[i] = wc[i].word
	}
	return ret
}

func readCSV(filePath string, separator rune, skipHeader bool) ([][]string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, errors.Wrap(err, "Can't open file")
	}
	defer f.Close()
	csvReader := csv.NewReader(f)
	csvReader.Comma = separator
	if skipHeader {
		_, err := csvReader.Read()
		if err != nil {
			return nil, errors.Wrap(err, "Can't skip header")
		}
	}
	records, err := csvReader.ReadAll()
	if err != nil {
		return nil, errors.Wrap(err, "Can't read file contents")
	}
	return records, nil
}
