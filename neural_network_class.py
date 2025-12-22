import jax.numpy as jnp
from jax import random
import yaml
import time

class FeedforwardNeuralNetwork:
    def __init__(self, configuration_file):
        self.supported_activation_functions = ['relu', 'softmax']
        self.key = random.key(time.time_ns())
        self.config = self._get_config(configuration_file)
        self.layers = self.config.get('layers')
        self._validate_structure()
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

    ## Get config file
    def _get_config(self, configuration_file):
        sample_config = """layers:
- neurons: 3
- neurons: 4
  activation: relu
- neurons: 2
  activation: softmax"""

        try:
            with open(configuration_file, 'r') as config_file:
                config = yaml.safe_load(config_file)
            return config
        except Exception as e:
            with open(configuration_file, 'w') as config_file:
                config_file.write(sample_config)
            print(f"An error occurred upon obtaining config file: {e}")
            print(f"{configuration_file} either doesn't exist or is inaccessible")
            print(f"A sample {configuration_file} has been created. Please edit it according to your needs")
            exit()

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
        if not isinstance(activation, jnp.ndarray) or not (jnp.issubdtype(activation.dtype, jnp.integer) or jnp.issubdtype(activation.dtype, jnp.floating)):
            self._handle_exception(TypeError("Input activation vector must be an integer or floating dtype jnumpy array"))
        elif jnp.any(jnp.isnan(activation)):
            self._handle_exception(ValueError("Input activation vector cannot contain NaN values"))
        elif activation.shape != (neurons, 1):
            self._handle_exception(ValueError(f"Input activation vector shape {activation.shape} needs to match the shape {neurons, 1}"))

    ## Initialization
    def _initialize_weights(self):
        weights = {}
        mean = 0
        for i in range(len(self.layers) - 1):
            self.key, subkey = random.split(self.key)
            standard_deviation = jnp.sqrt(2.0 / self.layers[i].get('neurons'))
            weights[f'layer{i}'] = random.normal(subkey, shape=(self.layers[i + 1].get('neurons'), self.layers[i].get('neurons')), dtype=jnp.float32) * standard_deviation + mean
            del subkey
        del self.key
        return weights

    def _initialize_biases(self):
        biases = {}
        for i in range(len(self.layers) - 1):
            neurons = self.layers[i + 1].get('neurons')
            biases[f'layer{i + 1}'] = jnp.zeros((neurons, 1), dtype=jnp.float32)
        return biases

    ## Activation functions
    def _relu(self, x):
        return jnp.maximum(0, x)

    def _softmax(self, x):
        exp_x = jnp.exp(x - jnp.max(x, axis=0, keepdims=True))
        return exp_x / jnp.sum(exp_x, axis=0, keepdims=True)

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
        return jnp.mean(jnp.square(y_true - y_pred))
    