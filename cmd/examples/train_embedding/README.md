# Example of how to use Embedding layers

Let's imagine that we have need to do following task:

    Convert the textual assessment of the quality of the work into a numerical grade.

    If work is not done (or is done in a bad way) then assessment is 0.0

    If work is done (and is done in a good way) then assessment is 1.0

    If work is done in a mediocre way then assessment is 0.5

Examples of input data (with corresponding assessment):
```
Well done! - 1.0
Good work - 1.0
Great effort - 1.0
nice work - 1.0
Excellent! - 1.0
Weak - 0.0
Poor effort! - 0.0
not good - 0.0
poor work - 0.0
Could be way better. - 0.0
average - 0.5
middle level - 0.5
ordinary stuff - 0.5
boilerplate - 0.5
standart approach - 0.5
```

Restrictions:
* Assume that vocabulary size is 50
* Max amount of words in a text assesment 5

So, neural network structure would be:
* Input shape equals to {**1** x **Max number of words in a sentence**}. Max number of words is defined as "5"
* Output shape equals to {**1** x **1**} since there is only one possible output (0.0, 1.0 or 0.5 in perfect conditions)
* Layers:
    * Embedding layer - Let's assume embedding dimensions is 12.

        It means that this layer has **Vocabulary size (which is 50)**-tensor of size **embedding dimensions (which is 12)**.
    
        No activation function is required. No bias is required.
    * Flatten layer - just represent N-tensor as [**1** x **Total number of elements in tensor**]
    * Output layer - just linear layer with sigmoid activation function.

        Number of input neurons: [**Max number of words in a sentence** x **Embedding dimensions**]. Number of output neurons: [**1** x **1**]
        
        No bias is required. 
* How to represent text as numerical data? Well, [HashingTrick](../../utils.go#L313) and [PaddingInt64Slice](../../utils.go#L288) will help to do this task

Final representation of network:
input(1, 5) -> embedding(inputs=5, voc=50, dims=12) -> flatten(5,12) -> linear(inputs=60, outputs=1) -> sigmoid(1)

Assume that number of training epochs is 200, learning rate is 0.01, solver is Adam, batch size is 1

Main code is in [main.go file](main.go). I guess it's pretty straightforward. But if it's not than I appreciate yours PR to improve this document

Simply execute:
```shell
go run main.go
```

Final output on for trainig data (may vary due the nature of rand() calls):
```shell
Text assessment: Weak
        Its hashed value: [5 0 0 0 0]
        Its defined numerical assessment: 0.0
        Its evaluated numerical assessment: 0.2
        Difference between defined and evaluated: 0.2
Text assessment: middle level
        Its hashed value: [12 40 0 0 0]
        Its defined numerical assessment: 0.5
        Its evaluated numerical assessment: 0.5
        Difference between defined and evaluated: 0.0
Text assessment: not good
        Its hashed value: [31 14 0 0 0]
        Its defined numerical assessment: 0.0
        Its evaluated numerical assessment: 0.1
        Difference between defined and evaluated: 0.1
Text assessment: Good work
        Its hashed value: [14 24 0 0 0]
        Its defined numerical assessment: 1.0
        Its evaluated numerical assessment: 0.9
        Difference between defined and evaluated: 0.1
Text assessment: ordinary stuff
        Its hashed value: [23 44 0 0 0]
        Its defined numerical assessment: 0.5
        Its evaluated numerical assessment: 0.5
        Difference between defined and evaluated: 0.0
Text assessment: Could be way better.
        Its hashed value: [36 25 5 18 0]
        Its defined numerical assessment: 0.0
        Its evaluated numerical assessment: 0.0
        Difference between defined and evaluated: 0.0
Text assessment: average
        Its hashed value: [35 0 0 0 0]
        Its defined numerical assessment: 0.5
        Its evaluated numerical assessment: 0.5
        Difference between defined and evaluated: 0.0
Text assessment: Great effort
        Its hashed value: [26 11 0 0 0]
        Its defined numerical assessment: 1.0
        Its evaluated numerical assessment: 1.0
        Difference between defined and evaluated: 0.0
Text assessment: poor work
        Its hashed value: [28 24 0 0 0]
        Its defined numerical assessment: 0.0
        Its evaluated numerical assessment: 0.1
        Difference between defined and evaluated: 0.1
Text assessment: boilerplate
        Its hashed value: [36 0 0 0 0]
        Its defined numerical assessment: 0.5
        Its evaluated numerical assessment: 0.5
        Difference between defined and evaluated: 0.0
Text assessment: standart approach
        Its hashed value: [13 29 0 0 0]
        Its defined numerical assessment: 0.5
        Its evaluated numerical assessment: 0.5
        Difference between defined and evaluated: 0.0
Text assessment: Poor effort!
        Its hashed value: [28 11 0 0 0]
        Its defined numerical assessment: 0.0
        Its evaluated numerical assessment: 0.0
        Difference between defined and evaluated: 0.0
Text assessment: Excellent!
        Its hashed value: [26 0 0 0 0]
        Its defined numerical assessment: 1.0
        Its evaluated numerical assessment: 0.9
        Difference between defined and evaluated: 0.1
Text assessment: nice work
        Its hashed value: [34 24 0 0 0]
        Its defined numerical assessment: 1.0
        Its evaluated numerical assessment: 1.0
        Difference between defined and evaluated: 0.0
Text assessment: Well done!
        Its hashed value: [26 13 0 0 0]
        Its defined numerical assessment: 1.0
        Its evaluated numerical assessment: 1.0
```