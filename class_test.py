import numpy as np
import pandas as pd
from neural_network_class import FeedforwardNeuralNetwork as fnn

# activation = np.array([
#     [1],
#     [2],
#     [3]
# ])

## Extraction

df = pd.read_csv('data.csv')

features_0 = df.iloc[:, 0].to_numpy() # Features extraction
features_4 = [] # Parsing and reshaping
for item in features_0:
    features_1 = str(item)
    features_2 = np.array([list(map(float, features_1.split(',')))])
    features_3 = features_2.reshape(-1, 1)
    features_4.append(features_3)

ground_truth_0 = df.iloc[:, 1].to_numpy() # Ground truth extraction
ground_truth_4 = [] # Parsing and reshaping
for item in ground_truth_0:
    ground_truth_1 = str(item)
    ground_truth_2 = np.array([list(map(float, ground_truth_1.split(',')))])
    ground_truth_3 = ground_truth_2.reshape(-1, 1)
    ground_truth_4.append(ground_truth_3)

# features_number/ground_truth_number: the number means the different stages of processing data from csv file


# Refer to todo_and_notes.md for more info on the below line
activation = np.array(features_4[0])


## Testing

neural_network = fnn('config.json')
output = neural_network.forward(activation) # With this line, forward pass can be executed multiple times each with different input activations while keeping weights and biases the same. Basically multiple instances of the neural network.
print(output)
print(np.sum(output))
