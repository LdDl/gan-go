# Example of how to use LSTM layer

This example is not about GAN itself: a small word level language model is trained to continue sentences of a tiny corpus.

Corpus:
```
the quick brown fox jumps over the lazy dog
the little cat sleeps under the warm sun
a small bird sings in the green garden
the old dog walks slowly through the park
a young fox hides behind the tall tree
```

Network structure:
```
input(4 word indices) => embedding(voc=31, dims=16) => lstm(inputs=16, hidden=32) => linear(31, 32) + softmax
```

Every training window is a sequence of 4 words and the target is the same sequence shifted by one word, so the network learns to predict the next word for every position of the window.

Training details: cross entropy loss, Adam solver, learning rate is 0.01, 200 epochs.

After training the network is asked to continue every sentence of the corpus given its first 4 words: the most probable word at the last position of the window is appended and the window slides forward.

Simply execute:
```shell
go run main.go
```

Final output (may vary due the nature of rand() calls):
```shell
Vocabulary size: 31
Number of training windows: 21
Epoch 0:
	Loss: 0.11104210591411041
Epoch 40:
	Loss: 0.00882740368445253
Epoch 80:
	Loss: 0.00631720821663664
Epoch 120:
	Loss: 2.4566094610844825e-05
Epoch 160:
	Loss: 0.011674595601638622
Epoch 200:
	Loss: 9.005437827311226e-06
Start testing generator after final epoch
	Seed: the quick brown fox
	Continued: the quick brown fox jumps over the lazy dog [OK]
	Seed: the little cat sleeps
	Continued: the little cat sleeps under the warm sun [OK]
	Seed: a small bird sings
	Continued: a small bird sings in the green garden [OK]
	Seed: the old dog walks
	Continued: the old dog walks slowly through the park [OK]
	Seed: a young fox hides
	Continued: a young fox hides behind the tall tree [OK]
```
