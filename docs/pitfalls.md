# Gorgonia pitfalls

Three non-obvious properties of Gorgonia (v0.9.x) discovered while developing this library. Every one of them produced silently wrong numbers rather than an error, so they are documented here in detail and guarded by tests.

## 1. Value nodes are deduplicated by name, not by value

Gorgonia hashes input (value) nodes by type, shape and name only. The value is NOT part of the hash. Two unnamed scalar nodes of the same dtype get the same hash, and the graph silently merges them into a single node, keeping the value of the first one.

The library hit this inside `HuberLoss`. The original code created three unnamed scalars: $\delta$, $\delta^2$ and $1.0$. All three collapsed into one node holding $\delta$, so instead of

$$\delta^2\left(\sqrt{1 + (x/\delta)^2} - 1\right) \tag{1}$$

the graph computed

$$\delta\left(\sqrt{\delta + (x/\delta)^2} - \delta\right) \tag{2}$$

For $\delta = 2$ and $x = \pm 0.2$ the loss was $-2.329$ instead of $+0.0199$, with no error anywhere.

Broken pattern:

```go
deltaScalar := gorgonia.NewScalar(g, dtype, gorgonia.WithValue(2.0))
oneScalar := gorgonia.NewScalar(g, dtype, gorgonia.WithValue(1.0))
// oneScalar IS deltaScalar now, its value is 2.0
```

Fixed pattern (see `HuberLoss` and `stableLog` in [loss.go](../loss.go)):

```go
deltaScalar := gorgonia.NewScalar(g, dtype, gorgonia.WithValue(2.0), gorgonia.WithName(fmt.Sprintf("huber_delta_%d_%d", a.ID(), b.ID())))
oneScalar := gorgonia.NewScalar(g, dtype, gorgonia.WithValue(1.0), gorgonia.WithName(fmt.Sprintf("huber_one_%d_%d", a.ID(), b.ID())))
```

Names are derived from IDs of the input nodes, so two loss calls on the same graph stay distinct as well. The same rule applies to automatically created zero states of recurrent layers. Guarded by `TestHuberLossSameGraph` in [loss_test.go](../loss_test.go).

## 2. Buffer reuse is blind to slice and reshape views

`gorgonia.Slice` and `gorgonia.Reshape` produce views: nodes sharing the backing memory of the source node. The tape machine is free to reuse the buffer of a node for the result of an operation once it considers the node dead, and the liveness analysis counts only direct graph edges. A pending read through a view is invisible to it.

The library hit this in the GRU layer. Hidden states of every time step were collected for the output as reshape views:

```go
outputs[t], _ = gorgonia.Reshape(hiddenState, tensor.Shape{1, batch, hiddenSize})
```

The next time step computes `Sub(hiddenState, candidate)`, which is the last direct consumer of `hiddenState`, so the tape machine wrote the subtraction result straight into its buffer. The stored view then pointed at that garbage: the final concatenated output contained $h_{prev} - n$ values instead of hidden states, while `FinalHidden()` was perfectly correct. LSTM and RNN escaped by pure structural luck: their last consumer of the hidden state is a matrix multiplication, which can not reuse a buffer of mismatched shape.

The fix materializes a standalone copy before storing (see the `outputAnchor` comment in [layer_gru.go](../layer_gru.go)):

```go
outputCopy, _ := gorgonia.Add(hiddenState, outputAnchor) // outputAnchor is a zero-valued node
outputs[t], _ = gorgonia.Reshape(outputCopy, tensor.Shape{1, batch, hiddenSize})
```

Related rule of thumb used across the recurrent layers: slice views are passed to unary operations only. A binary operation may write its result into the buffer of an operand, and when the operand is a view, that write corrupts the source tensor. Guarded by forward tests of all recurrent layers against references computed with plain Go loops.

## 3. Slice, activation, reshape: the order changes gradients

Extracting a gate of a recurrent layer requires three steps: slice the columns, reshape (a slicing range of width 1 collapses the dimension), apply the activation. The order of the last two steps changes the backward pass:

```go
// correct: slice, reshape, activate
gate, _ := gorgonia.Slice(gates, nil, gorgonia.S(from, to))
reshaped, _ := gorgonia.Reshape(gate, tensor.Shape{batch, hiddenSize})
return activation(reshaped)

// broken: slice, activate, reshape
gate, _ := gorgonia.Slice(gates, nil, gorgonia.S(from, to))
activated, _ := activation(gate)
return gorgonia.Reshape(activated, tensor.Shape{batch, hiddenSize})
```

Both variants produce identical forward values. The broken variant produces wrong analytic gradients: in a minimal reproduction with two sigmoid gates multiplied element wise, analytic $\partial L / \partial W$ came out as $0.0042$ against the numeric $0.0739$ for the same weight. The forward-only tests can not catch this class of bugs at all.

The correct order is encapsulated in the `sliceGate` helper of [layer.go](../layer.go). The whole class is guarded by numerical gradient checks (central finite differences against analytic gradients) for every recurrent layer in [layer_gradients_test.go](../layer_gradients_test.go).

One more detail of those checks worth knowing: the tape machine accumulates gradients over `RunAll` calls and `Reset` does not zero them. Analytic gradients of ALL learnables must be snapshotted right after the first run, before any finite difference runs.
