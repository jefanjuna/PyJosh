import numpy as np

class FeedforwardNeuralNetwork:
    def __init__(self, input_array):
        self.supported_activation_functions = ['relu', 'softmax']
        self.config = self._get_config(input_array)
        self.layers = self.config.get('layers')
        self._validate_structure()
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

    ## Get config file
    def _get_config(self, input_array):
        return input_array

    ## Validation
    def _handle_exception(self, exception):
        print(f"{type(exception).__name__}: {exception}")
        print("If you think that your config file is corrupted or don't know how to fix it, just delete the config file and a new random sample one will be generated!")
        exit()

    def _validate_structure(self):
        if not self.layers or not isinstance(self.layers, list):
            self._handle_exception(ValueError("Configuration must contain a non-empty 'layers' list"))

        input_neurons = self.layers[0].get('neurons') # First layer neuron validation
        if not isinstance(input_neurons, int) or input_neurons <= 0:
            self._handle_exception(ValueError("layer 0 must have a 'neurons' key with a positive integer value"))

        layers = self.layers[1:] # Exclude the first layer from validation specifically activation validation
        for index, layer in enumerate(layers, start=1):
            neurons = layer.get('neurons')
            activation_function = layer.get('activation')
            if not isinstance(neurons, int) or neurons <= 0:
                self._handle_exception(ValueError(f"layer {index} must have a 'neurons' key with a positive integer value"))
            elif activation_function is None or activation_function == '':
                self._handle_exception(ValueError(f"layer {index} must have an 'activation' key with a non-empty string value"))
            elif activation_function not in self.supported_activation_functions:
                self._handle_exception(ValueError(f"layer {index} has an unsupported activation function: {activation_function}. Supported activation functions are: {self.supported_activation_functions}"))

    def _validate_input(self, activation):
        neurons = self.layers[0].get('neurons')
        if not isinstance(activation, np.ndarray) or not (np.issubdtype(activation.dtype, np.integer) or np.issubdtype(activation.dtype, np.floating)):
            self._handle_exception(TypeError("Input activation vector must be an integer or floating dtype numpy array"))
        elif np.any(np.isnan(activation)):
            self._handle_exception(ValueError("Input activation vector cannot contain NaN values"))
        elif activation.shape != (neurons, 1):
            self._handle_exception(ValueError(f"Input activation vector shape {activation.shape} needs to match the shape {neurons, 1}"))

    ## Initialization
    def _initialize_weights(self):
        weights = {}
        for i in range(len(self.layers) - 1):
            standard_deviation = np.sqrt(2.0 / self.layers[i].get('neurons'))
            weights[f'layer{i}'] = np.random.normal(0, standard_deviation, size=(self.layers[i + 1].get('neurons'), self.layers[i].get('neurons')))
        return weights

    def _initialize_biases(self):
        biases = {}
        for i in range(len(self.layers) - 1):
            neurons = self.layers[i + 1].get('neurons')
            biases[f'layer{i + 1}'] = np.zeros((neurons, 1))
        return biases

    ## Activation functions
    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    ## Forward pass
    def forward(self, activation):
        self._validate_input(activation)
        activation_function_map = {
            'relu': self._relu,
            'softmax': self._softmax
        }
        activationsss = []
        for i in range(len(self.layers) - 1):
            z = self.weights[f'layer{i}'] @ activation + self.biases[f'layer{i + 1}']
            activation_function = activation_function_map.get(self.layers[i + 1].get('activation'))
            activation = activation_function(z)
            activationsss.append(activation)
        print(activationsss)
        return activation

    ## Loss functions
    def _mean_squared_error(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

## Example usage
config = {
  "layers": [
    {
      "neurons": 3
    },
    {
      "neurons": 4,
      "activation": "relu"
    },
    {
      "neurons": 2,
      "activation": "softmax"
    }
  ]
}

activation = np.array([
    [1],
    [2],
    [3]
])

neural_network = FeedforwardNeuralNetwork(config)
output = neural_network.forward(activation) # With this line, forward pass can be executed multiple times each with different input activations while keeping weights and biases the same.
print(output)
print(np.sum(output))