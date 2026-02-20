import jax
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

        self.params = self._initialize_parameters()

    # Config

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
            print(f"Error loading config: {e}")
            print("Sample config created. Edit and rerun.")
            exit()

    def _validate_structure(self):
        if not self.layers or not isinstance(self.layers, list):
            raise ValueError("Config must contain a non-empty 'layers' list")

        for i, layer in enumerate(self.layers):
            neurons = layer.get('neurons')
            if not isinstance(neurons, int) or neurons <= 0:
                raise ValueError(f"Layer {i} must have positive integer neurons")

            if i > 0:
                activation = layer.get('activation')
                if activation not in self.supported_activation_functions:
                    raise ValueError(
                        f"Layer {i} activation must be one of {self.supported_activation_functions}"
                    )

    # Initialization

    def _initialize_parameters(self):
        params = []

        for i in range(len(self.layers) - 1):
            self.key, subkey = random.split(self.key)

            in_dim = self.layers[i]['neurons']
            out_dim = self.layers[i + 1]['neurons']

            std = jnp.sqrt(2.0 / in_dim)

            W = random.normal(subkey, (out_dim, in_dim)) * std
            b = jnp.zeros((out_dim,))

            params.append({"W": W, "b": b})

        return params

    # Activations

    def _relu(self, x):
        return jnp.maximum(0, x)

    def _softmax(self, x):
        x = x - jnp.max(x, axis=-1, keepdims=True)
        exp_x = jnp.exp(x)
        return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

    def _apply_activation(self, x, name):
        if name == "relu":
            return self._relu(x)
        elif name == "softmax":
            return self._softmax(x)

    # Forward

    def forward_with_params(self, params, x):
        """
        x shape:
        - single sample: (input_dim,)
        - batch: (batch_size, input_dim)
        """

        for i, layer in enumerate(params):
            W = layer["W"]
            b = layer["b"]

            x = jnp.dot(x, W.T) + b
            activation_name = self.layers[i + 1]['activation']
            x = self._apply_activation(x, activation_name)

        return x

    # User-facing forward
    def forward(self, x):
        return self.forward_with_params(self.params, x)

    # Loss

    def cross_entropy_loss(self, params, x, y):
        """
        x: (batch_size, input_dim)
        y: (batch_size, output_dim)
        """
        preds = self.forward_with_params(params, x)
        return -jnp.mean(jnp.sum(y * jnp.log(preds + 1e-8), axis=-1))

    # Training Step

    def train_step(self, params, x, y, lr):

        loss_value, grads = jax.value_and_grad(self.cross_entropy_loss)(
            params, x, y
        )

        new_params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g,
            params,
            grads
        )

        return new_params, loss_value

    # Training Loop

    def train(self, x, y, epochs=1000, lr=0.01):

        params = self.params

        for epoch in range(epochs):
            params, loss = self.train_step(params, x, y, lr)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        self.params = params
