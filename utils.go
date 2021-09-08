package gan_go

import (
	"crypto/md5"
	"crypto/sha256"
	"crypto/sha512"
	"fmt"
	"hash/fnv"
	"image/color"
	"math/big"
	"math/rand"
	"sort"
	"strings"

	"regexp"

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

// GenerateNormTestSamples Generates test samples for provided arguments for [Normal distribution]
//
// vmGenerator - tape machine used for GAN
// vmDiscriminator - tape machined used for Discriminator only
// inputGenerator - node for holding value of Generator's input
// inputDiscriminator - node for holding value of Discriminator's input
// graphValue - variable with access to Generator's output
// numSamples - how many sample generate
// batchSize - batch size basically
// n - number of elements in each batch (latent space size)
//
func GenerateNormTestSamples(vmGenerator, vmDiscriminator gorgonia.VM, inputGenerator, inputDiscriminator *gorgonia.Node, generatorOutValue gorgonia.Value, numSamples, batchSize, n int, shape tensor.Shape) (*tensor.Dense, error) {
	var testSamplesTensor *tensor.Dense

	for i := 0; i < numSamples; i++ {
		latentSpaceSamples := NormRandDense(batchSize, n)
		if len(shape) > 0 {
			err := latentSpaceSamples.Reshape(shape...)
			if err != nil {
				return nil, errors.Wrap(err, "Can't reshape latent spaces")
			}
		}
		err := gorgonia.Let(inputGenerator, latentSpaceSamples)
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

// GenerateUniformTestSamples Generates test samples for provided arguments [Uniform distribution]
//
// vmGenerator - tape machine used for GAN
// vmDiscriminator - tape machined used for Discriminator only
// inputGenerator - node for holding value of Generator's input
// inputDiscriminator - node for holding value of Discriminator's input
// graphValue - variable with access to Generator's output
// numSamples - how many sample generate
// batchSize - batch size basically
// n - number of elements in each batch (latent space size)
//
func GenerateUniformTestSamples(vmGenerator, vmDiscriminator gorgonia.VM, inputGenerator, inputDiscriminator *gorgonia.Node, generatorOutValue gorgonia.Value, numSamples, batchSize, n int, shape tensor.Shape) (*tensor.Dense, error) {
	var testSamplesTensor *tensor.Dense

	for i := 0; i < numSamples; i++ {
		latentSpaceSamples := UniformRandDense(batchSize, n)
		if len(shape) > 0 {
			err := latentSpaceSamples.Reshape(shape...)
			if err != nil {
				return nil, errors.Wrap(err, "Can't reshape latent spaces")
			}
		}
		err := gorgonia.Let(inputGenerator, latentSpaceSamples)
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

func OneHotEncode(sl []string) ([][]int, error) {
	result := [][]int{}
	unique := make(map[string]bool)
	for _, s := range sl {
		unique[s] = true
	}
	uniqueSlice := make([]string, 0, len(unique))
	for k := range unique {
		uniqueSlice = append(uniqueSlice, k)
	}
	sort.Strings(uniqueSlice)
	maxIdx := len(uniqueSlice)
	for i := range sl {
		oneHotEncodedResult := make([]int, maxIdx)
		oneHotIdx := findIdxStrings(sl[i], uniqueSlice)
		if oneHotIdx == -1 {
			return nil, fmt.Errorf("Index went to -1. This should not happen at all")
		}
		oneHotEncodedResult[oneHotIdx] = 1
		result = append(result, oneHotEncodedResult)
	}
	return result, nil
}

func findIdxStrings(s string, slice []string) int {
	for i, item := range slice {
		if item == s {
			return i
		}
	}
	return -1
}

type PaddingSliceType int

const (
	PADDING_PRE = PaddingSliceType(iota)
	PADDING_POST
)

// PaddingInt64Slice Append (or prepend) zero-based elements to slice until max length is reached.
// If defined length is less or equal to length of provided slice then provided slice will be returned
// Inspired by: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
func PaddingInt64Slice(sl []int64, maxLen int, pt PaddingSliceType) []int64 {
	if maxLen <= len(sl) {
		return sl
	}
	newSL := make([]int64, maxLen-len(sl))
	switch pt {
	case PADDING_POST:
		return append(sl, newSL...)
	case PADDING_PRE:
		return append(newSL, sl...)
	default:
		return sl
	}
}

type HashType int

const (
	HASH_FNV32A = HashType(iota + 1)
	HASH_FNV64A
	HASH_SHA256
	HASH_SHA512
	HASH_MD5
)

func HashingTrick(sentence string, vocab int, ht HashType) ([]int64, error) {
	// Split sentence into words
	regexpStr := `[^\s!,.?":;0-9]+`
	reg, err := regexp.Compile(`[^\s!,.?":;0-9]+`)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("Can't compile regexp string: '%s'", regexpStr))
	}
	strsRepresentation := reg.FindAllString(sentence, -1)
	for i := range strsRepresentation {
		strsRepresentation[i] = strings.ToLower(strsRepresentation[i])
	}
	// Apply hashing function
	switch ht {
	case HASH_FNV32A:
		return HashingTrickFNV32A(strsRepresentation, vocab), nil
	case HASH_FNV64A:
		return HashingTrickFNV64A(strsRepresentation, vocab), nil
	case HASH_SHA256:
		return HashingTrickSHA256(strsRepresentation, vocab)
	case HASH_SHA512:
		return HashingTrickSHA512(strsRepresentation, vocab)
	case HASH_MD5:
		return HashingTrickMD5(strsRepresentation, vocab)
	default:
		return nil, fmt.Errorf("hash type of '%d' is not handled yet", ht)
	}
}

func HashingTrickFNV32A(sentenceWords []string, vocab int) []int64 {
	hashedList := make([]int, vocab)
	ans := make([]int64, len(sentenceWords))
	for i, word := range sentenceWords {
		h := fnv.New32a()
		h.Write([]byte(word))
		hashed := h.Sum32()
		hexInt := big.NewInt(int64(hashed))
		bigVectorLength := big.NewInt(int64(vocab))
		modulo := new(big.Int)
		modulo = modulo.Mod(hexInt, bigVectorLength)
		moduloInt64 := modulo.Int64()
		hashedList[moduloInt64] += 1
		if hashedList[moduloInt64] > 0 {
			ans[i] = moduloInt64
		}
	}
	return ans
}

func HashingTrickFNV64A(sentenceWords []string, vocab int) []int64 {
	hashedList := make([]int, vocab)
	ans := make([]int64, len(sentenceWords))
	for i, word := range sentenceWords {
		h := fnv.New64a()
		h.Write([]byte(word))
		hashed := h.Sum64()
		hexInt := big.NewInt(int64(hashed))
		bigVectorLength := big.NewInt(int64(vocab))
		modulo := new(big.Int)
		modulo = modulo.Mod(hexInt, bigVectorLength)
		moduloInt64 := modulo.Int64()
		hashedList[moduloInt64] += 1
		if hashedList[moduloInt64] > 0 {
			ans[i] = moduloInt64
		}
	}
	return ans
}

func HashingTrickSHA256(sentenceWords []string, vocab int) ([]int64, error) {
	hashedList := make([]int, vocab)
	ans := make([]int64, len(sentenceWords))
	for i, word := range sentenceWords {
		hashedValue := sha256.New()
		hashedValue.Write([]byte(word))
		hexStr := fmt.Sprintf("%x", hashedValue.Sum(nil))
		hexInt := new(big.Int)
		hexInt, ok := hexInt.SetString(hexStr, 16)
		if !ok {
			return nil, fmt.Errorf("can't create big int from hex")
		}
		bigVectorLength := big.NewInt(int64(vocab))
		modulo := new(big.Int)
		modulo = modulo.Mod(hexInt, bigVectorLength)
		moduloInt64 := modulo.Int64()
		hashedList[moduloInt64] += 1
		if hashedList[moduloInt64] > 0 {
			ans[i] = moduloInt64
		}
	}
	return ans, nil
}

func HashingTrickSHA512(sentenceWords []string, vocab int) ([]int64, error) {
	hashedList := make([]int, vocab)
	ans := make([]int64, len(sentenceWords))
	for i, word := range sentenceWords {
		hashedValue := sha512.New()
		hashedValue.Write([]byte(word))
		hexStr := fmt.Sprintf("%x", hashedValue.Sum(nil))
		hexInt := new(big.Int)
		hexInt, ok := hexInt.SetString(hexStr, 16)
		if !ok {
			return nil, fmt.Errorf("can't create big int from hex")
		}
		bigVectorLength := big.NewInt(int64(vocab))
		modulo := new(big.Int)
		modulo = modulo.Mod(hexInt, bigVectorLength)
		moduloInt64 := modulo.Int64()
		hashedList[moduloInt64] += 1
		if hashedList[moduloInt64] > 0 {
			ans[i] = moduloInt64
		}
	}
	return ans, nil
}

func HashingTrickMD5(sentenceWords []string, vocab int) ([]int64, error) {
	hashedList := make([]int, vocab)
	ans := make([]int64, len(sentenceWords))
	for i, word := range sentenceWords {
		hashedValue := md5.New()
		hashedValue.Write([]byte(word))
		hexStr := fmt.Sprintf("%x", hashedValue.Sum(nil))
		hexInt := new(big.Int)
		hexInt, ok := hexInt.SetString(hexStr, 16)
		if !ok {
			return nil, fmt.Errorf("can't create big int from hex")
		}
		bigVectorLength := big.NewInt(int64(vocab))
		modulo := new(big.Int)
		modulo = modulo.Mod(hexInt, bigVectorLength)
		moduloInt64 := modulo.Int64()
		hashedList[moduloInt64] += 1
		if hashedList[moduloInt64] > 0 {
			ans[i] = moduloInt64
		}
	}
	return ans, nil
}
