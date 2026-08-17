# Example of how to use convolutional and maxpool layers

This example is not about GAN itself: here a simple CNN classifier is trained to recognize three hardcoded 9x8 binary images of characters: 'X', 'T' and 'O'.

Each character is represented as {**1** x **1** x **9** x **8**} tensor (NCHW). E.g. 'X' looks like:
```
0 0 0 0 0 0 0 0
0 1 0 0 0 1 0 0
0 0 1 0 1 0 0 0
0 0 1 0 1 0 0 0
0 0 0 1 0 0 0 0
0 0 1 0 1 0 0 0
0 0 1 0 1 0 0 0
0 1 0 0 0 1 0 0
0 1 0 0 0 1 0 0
```

Network structure:
```
input(9,8) => filters=5,size=3x3,conv(7,6) + ReLU => dropout(0.3) => filters=5,size=2x2,maxpool(3,3) => 5*flatten(9) => linear(3, 45) + sigmoid
```

Training details: 1000 training steps for each character, MSE loss, RMSProp solver, learning rate is 0.01.

After training the network is tested both on training data and on noisy versions of the same images (random noise is added to non-zero pixels).

Simply execute:
```shell
go run main.go
```

Final output (may vary due the nature of rand() calls):
```shell
X => Should give [1, 0, 0] R[   0.9999908039116123  3.837868975784883e-07  2.775051668886621e-06]
	noisy X => Should give [1, 0, 0] R[    0.9999342949415082   6.495541204950894e-06  3.6610997050788296e-05]
T => Should give [0, 1, 0] R[3.4951088432945394e-06      0.9999960313832091  1.4535406340201251e-07]
	noisy T => Should give [0, 1, 0] R[2.5626526151386557e-06      0.9999938890216405   8.237476440989576e-08]
O => Should give [0, 0, 1] R[4.492867565213136e-06  5.427829062329569e-07     0.9999970744490388]
	noisy O => Should give [0, 0, 1] R[1.3759089837489314e-06   4.235235799855587e-07      0.9999977295036324]
```
