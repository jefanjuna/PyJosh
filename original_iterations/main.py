## AI Description:
## This is a simple feedforward neural network with 3 input neurons, 4 hidden neurons, and 2 output neurons.
## The input layer is fully connected to the hidden layer, and the hidden layer is fully connected to the output layer.
## The activation function used is ReLU (Rectified Linear Unit).
## The weights are initialized using He normal initialization, which is suitable for ReLU activation functions.
## The biases are initialized to 0.
## The input to the network is a matrix
## The output of the network is a matrix (vector) after applying the ReLU activation function to the weighted sum of the inputs.
## The output is printed as a matrix/vector to the console.

import numpy as np

## Note: the only thing here that can be modified is the neurons dict and the input activation vector

## numbered layers: the first and last layers are input and output layers respectively
## please do not change the naming convention of the layers
## This dict can be modified accordingly to add more layers or neurons
neurons = {
    'layer0': 3,
    'layer1': 4,
    'layer2': 2
}

## Input layer vector
## Note this is a vector and needs to match the number of neurons specified above in layer0 in the convention: layer0: n, vector shape: (n, 1)
## Since this is the input, the values here are arbitary
activation = np.array([
    [1],
    [1],
    [1]
])

assert activation.shape == (neurons['layer0'], 1), f"Input activation vector shape {activation.shape} needs to match the shape {neurons['layer0'], 1}"

## Initialization
## He normal weights initialization
## first layer of weights means taking input of first layer of neurons
def initialize_weights():
    weights = {}
    for i in range(len(neurons) - 1):
        standard_deviation = np.sqrt(2.0 / neurons[f'layer{i}'])
        weights[f'layer{i}'] = np.random.normal(0, standard_deviation, size=(neurons[f'layer{i + 1}'], neurons[f'layer{i}']))
    return weights

weights = initialize_weights() ## initialize only one time

## Initialize biases as 0
def initialize_biases():
    biases = {}
    for i in range(len(neurons) - 1):
        biases[f'layer{i + 1}'] = np.zeros((neurons[f'layer{i + 1}'], 1))
    return biases

biases = initialize_biases() ## initialize only one time

## ReLU activation function
def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    # Subtracting the max value along each row (axis=0) for numerical stability and to prevent overflow
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    # Dividing by the sum of exponentials
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

## Output
for i in range(len(neurons) - 1):
    z = weights[f'layer{i}'] @ activation + biases[f'layer{i + 1}']
    if i < len(neurons) - 2:
        activation = ReLU(z)
    else:
        activation = softmax(z)
        ## The activation function here applied to the last layer can be nothing for regression, or sigmoig for binary classification, or softmax for multiclass classification
        ## In this case, we are just applying softmax to the last layer since this neural network can be expanded to the user's needs

## Basically same as: activation = ReLU(np.matmul(weights[f'layer{i}'], activation) + biases[f'layer{i + 1}'])
## the @ operator is just a shorthand for np.matmul
## Use np.matmul instead of np.dot for this usecase

print(activation)
print(np.sum(activation, axis=0, keepdims=True))
