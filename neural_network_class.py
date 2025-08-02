import numpy as np
import json

class FeedforwardNeuralNetwork:
    def __init__(self, configuration_file):
        self.config = self._get_config(configuration_file)
        self.layers = self.config.get('layers')
        self._validate_structure()
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

    ## Get config file
    def _get_config(self, configuration_file):
        with open(configuration_file, 'r') as config_file:
            config = json.load(config_file)
        return config

    ## Validation
    def _validate_structure(self):
        if not self.layers or not isinstance(self.layers, list):
            raise ValueError("Configuration must contain a non-empty 'layers' list.")
            
        supported_activation_functions = ['relu', 'softmax']
        layers = self.layers[1:] # Exclude the first layer for activation validation
        for index, layer in enumerate(layers, start=1):
            neurons = layer.get('neurons')
            activation_function = layer.get('activation')
            if not isinstance(neurons, int) or neurons <= 0:
                raise ValueError(f"layer {index} must have a 'neurons' key with a positive integer value")
            elif activation_function is None or activation_function == '':
                raise ValueError(f"layer {index} must have an 'activation' key with a non-empty value")
            elif activation_function not in supported_activation_functions:
                raise ValueError(f"layer {index} has unsupported activation function: {activation_function}. Supported activation functions are: {supported_activation_functions}")

    def _validate_input(self, activation):
        neurons = self.layers[0].get('neurons')
        if not isinstance(activation, np.ndarray) or not (np.issubdtype(activation.dtype, np.integer) or np.issubdtype(activation.dtype, np.floating)):
            raise TypeError("Input activation vector must be a integer or floating dtype numpy array")
        elif np.any(np.isnan(activation)):
            raise ValueError("Input activation vector cannot contain NaN values")
        elif activation.shape != (neurons, 1):
            raise ValueError(f"Input activation vector shape {activation.shape} needs to match the shape {neurons, 1}")

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
        for i in range(len(self.layers) - 1):
            z = self.weights[f'layer{i}'] @ activation + self.biases[f'layer{i + 1}']
            activation_function = activation_function_map.get(self.layers[i + 1].get('activation'))
            activation = activation_function(z)
        return activation
    
    ## Loss functions
    def _mean_squared_error(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    