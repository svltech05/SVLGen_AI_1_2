Deep learning :

ANN : Artificial Neural Network

Neuron : is a collection of arrays
collection of such neurons is known as a neural network.

NN : will be having 3 layers

1. input layer
2. hidden layer
3. output layer
Each layer will consists of multiple neurons.

Input layer : Takes data from data source (like : text file, json file, csv file
database etc...), and it passes that input along with weights to the hidden layer.
Dot product (matrix multiplication) of the input matrix and weights will be sent to
hidden layer.

Hidden layer : it takes data from input layer (previous layer)

input : X.dot(w) (X input matrix * w (weight matrix)

on this input, the activation function will be applied :

	z = f(x.dot(w))

this value will be sent to output layer along with weights (w2)

Output layer : Take input from hidden layer

	l = z.dot(w2) (input for the output layer)

for this activation function will be applied :
      	y = f(l)

This is the final output of the network.