import jax.numpy as jnp
import pandas as pd
from neural_network_class import FeedforwardNeuralNetwork as fnn

# Data Pipeline

df = pd.read_csv('data.csv')

# Features
features_raw = df.iloc[:, 0].to_numpy()

features_list = []
for item in features_raw:
    values = list(map(float, str(item).split(',')))
    features_list.append(jnp.array(values))  # shape: (input_dim,)

features = jnp.stack(features_list)  # shape: (batch_size, input_dim)

# Ground Truth
ground_truth_raw = df.iloc[:, 1].to_numpy()

ground_truth_list = []
for item in ground_truth_raw:
    values = list(map(float, str(item).split(',')))
    ground_truth_list.append(jnp.array(values))  # shape: (output_dim,)

ground_truth = jnp.stack(ground_truth_list)  # shape: (batch_size, output_dim)

print("Features shape:", features.shape)
print("Ground truth shape:", ground_truth.shape)

# Initialize Network

neural_network = fnn('config.yml')

# Train

neural_network.train(
    features,
    ground_truth,
    epochs=1000,
    lr=0.01
)

# Inference

# Single sample (keep batch dimension)
sample = features[0:1]   # shape: (1, input_dim)

output = neural_network.forward(sample)

print("Prediction:", output)
print("Sum of probabilities:", jnp.sum(output))
