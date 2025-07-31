import numpy as np

class FeedforwardNeuralNetwork:
    def __init__(self, neurons):
        self.neurons = neurons
        self._validate_structure()
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

    ## Find number of neurons in a layer
    def _get_neurons(self, layer):
        neurons = list(self.neurons.values())[layer]
        return neurons

    ## Validation
    def _validate_structure(self):
        for layer, number_of_neurons in self.neurons.items():
            assert isinstance(number_of_neurons, int) and number_of_neurons > 0, f"{layer} must have a positive integer number of neurons, but got {number_of_neurons}"

    def _validate_input(self, input_activation):
        self.activation = input_activation
        neurons = self._get_neurons(0)
        assert self.activation.shape == (neurons, 1), f"Input activation vector shape {self.activation.shape} needs to match the shape {neurons, 1}"

    ## Initialization
    def _initialize_weights(self):
        weights = {}
        for i in range(len(self.neurons) - 1):
            standard_deviation = np.sqrt(2.0 / self._get_neurons(i))
            weights[f'layer{i}'] = np.random.normal(0, standard_deviation, size=(self._get_neurons(i + 1), self._get_neurons(i)))
        return weights

    def _initialize_biases(self):
        biases = {}
        for i in range(len(self.neurons) - 1):
            neurons = self._get_neurons(i + 1)
            biases[f'layer{i + 1}'] = np.zeros((neurons, 1))
        return biases

    ## Activation functions
    def _ReLU(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    ## Forward pass
    def forward(self, input_activation):
        self.activation = input_activation
        self._validate_input(self.activation)
        for i in range(len(self.neurons) - 1):
            z = self.weights[f'layer{i}'] @ self.activation + self.biases[f'layer{i + 1}']
            if i < len(self.neurons) - 2:
                self.activation = self._ReLU(z)
            else:
                self.activation = self._softmax(z)
        return self.activation
    
    ## Loss functions
    def _mean_squared_error(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

## Example usage
neurons = {
    'input': 3,
    'hidden': 4,
    'output': 2
}

activation = np.array([
    [1],
    [2],
    [3]
])

neural_network = FeedforwardNeuralNetwork(neurons)
output = neural_network.forward(activation) # With this line, forward pass can be executed multiple times each with different input activations while keeping weights and biases the same.
print(output)
print(np.sum(output))