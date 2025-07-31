import numpy as np

## Network Structure
neurons = {
    'layer0': 3,
    'layer1': 4,
    'layer2': 2
}

for layer, number_of_neurons in neurons.items():
    assert isinstance(number_of_neurons, int) and number_of_neurons > 0, f"{layer} must have a positive integer number of neurons, but got {number_of_neurons}"

## Input
activation = np.array([
    [1],
    [2],
    [3]
])

assert activation.shape == (neurons['layer0'], 1), f"Input activation vector shape {activation.shape} needs to match the shape {neurons['layer0'], 1}"

## Initialization
def initialize_weights():
    weights = {}
    for i in range(len(neurons) - 1):
        standard_deviation = np.sqrt(2.0 / neurons[f'layer{i}'])
        weights[f'layer{i}'] = np.random.normal(0, standard_deviation, size=(neurons[f'layer{i + 1}'], neurons[f'layer{i}']))
    return weights

def initialize_biases():
    biases = {}
    for i in range(len(neurons) - 1):
        biases[f'layer{i + 1}'] = np.zeros((neurons[f'layer{i + 1}'], 1))
    return biases

weights = initialize_weights()
biases = initialize_biases()

## Activation functions
def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

## Output
for i in range(len(neurons) - 1):
    z = weights[f'layer{i}'] @ activation + biases[f'layer{i + 1}']
    if i < len(neurons) - 2:
        activation = ReLU(z)
    else:
        activation = softmax(z)

print(activation)