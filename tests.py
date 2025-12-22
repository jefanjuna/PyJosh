# # import numpy as np
# # import pandas as pd

# # array0 = np.array([1, 2, 3])
# # array1 = np.array([
# #     [4],
# #     [5],
# #     [6]
# #     ])
# # array2 = np.array([
# #     [10],
# #     [16.5764763567476],
# #     [0]
# #     ])
# # array3 = np.array([
# #     [0],
# #     [0],
# #     [0]
# #     ])
# # array4 = np.array(
# #     [[1, 1, 1]]
# # )


# # print(np.square(2))
# # print(np.square(array0))
# # print(np.square(array1))
# # print(np.mean(array0))
# # print(np.mean(array1))
# # print(array2 - array1)
# # print(array1 / array1)
# # print(array3 / array3)
# # print(array2 / array1)



# # #batch_size = 2
# # df = pd.read_csv('data.csv')
# # ## Features extraction & parsing
# # features_0 = df.iloc[:, 0].to_numpy() # Extraction
# # #Parsing and reshaping
# # features_4 = []
# # for item in features_0:
# #     features_1 = str(item)
# #     features_2 = np.array([list(map(float, features_1.split(',')))])
# #     features_3 = features_2.reshape(-1, 1)
# #     features_4.append(features_3)

# # ## features_number the number means the different stages of processing features from csv file

# # print(features_4)
# # print(features_4[0])
# # print(features_4[0] @ array4)

# # ## same thing here for ground truth
# # ground_truth_0 = df.iloc[:, 1].to_numpy() # Extraction
# # #Parsing and reshaping
# # ground_truth_4 = []
# # for item in ground_truth_0:
# #     ground_truth_1 = str(item)
# #     ground_truth_2 = np.array([list(map(float, ground_truth_1.split(',')))])
# #     ground_truth_3 = ground_truth_2.reshape(-1, 1)
# #     ground_truth_4.append(ground_truth_3)

# # print(ground_truth_4)
# # print(ground_truth_4[2])


# import json

# with open('config.json', 'r') as config_file:
#     config = json.load(config_file)

# # print(config)
# # print("")
# # print(config.get('layers')[0])
# # print("")
# # print(config.get('layers')[0].get('neurons'))
# # print("")

# # neurons = []

# # for layer in config.get('layers'):
# #     neurons.append(layer.get('neurons'))

# # print(neurons)
# # print(neurons[0])

# # print("")
# # enumerated_neurons = enumerate(neurons)
# # print(list(enumerated_neurons))

# # print("")
# # print(config.get('layers')[0].get('neurons'))
# # print(config.get('layers')[2].get('activation'))

# # print("")
# # print(len(config.get('layers')))


# # # self.config.get('layers')[i].get('neurons')
# # # self.config.get('layers')[i].get('activation')
# # # len(self.config.get('layers'))

# # print("")
# # print(list(enumerate(config.get('layers'))))

# # print("")
# # print(config.get('layers'))

# print(config.get('layers')[0].get('activation'))

# print(config.get('layers')[0])

# for index, layer in enumerate(config.get('layers'), start=1):
#             neurons = layer.get('neurons')
#             activation_function = layer.get('activation')
#             print(activation_function)

# from jax import grad

# def f(x):
#     return x**2 + 3*x + 2

# df_dx = grad(f)

# print(df_dx(2.0))

# import numpy as np
# z = np.random.normal(0, 1, size=100000)
# x = 3.32983 * z
# print(np.std(z))         # ~1
# print(np.std(x))         # ~3.32983


# from jax import random
# key = random.key(42)
# print(random.normal(key))
# for i in range(10):
#   key, subkey = random.split(key)

#   val = random.normal(subkey)
#   del subkey  # The subkey is consumed by normal().

#   print(f"draw {i}: {val}")

# print(random.normal(key))


# import jax.numpy as jnp
# from jax import random

# key = random.key(42)
# stdev = 3.696522
# mean = 0

# for i in range(3):
#     key, subkey = random.split(key)
#     print(random.normal(subkey, dtype=jnp.float32, shape=(4, 3)) * stdev + mean)
#     del subkey
# del key


# a = jnp.array([[1, 2], [3, 4]])
# print(a * 5)   # Multiplies every element by 5
# print(a + 10)  # Adds 10 to every element 

# import jax.numpy as jnp
# from jax import random
# import time

# key = random.key(time.time_ns())

# supported_activation_functions = ['relu', 'softmax']
# num_of_activations = len(supported_activation_functions)

# for i in range(10):
#     key, subkey = random.split(key)
#     index = random.randint(subkey, shape=(), minval=0, maxval=num_of_activations, dtype=jnp.int32)
#     print(supported_activation_functions[index])
#     del subkey
# del key

# for i in range(3):
#     print(i)

# import time
# print(time.time_ns())

# print("""layers:
# - neurons: 3 # layer 1
# - neurons: 4 # layer 2, you can add however many layers you want. Number of neurons is changeable.
#   activation: relu # Don't forget to mention activation. First layer naturally has no activation; if you add it, it'll be ignored.
# - neurons: 2 # layer 3
#   activation: softmax
# """)

# print("""
#       layers:
#       - neurons: 3 # layer 1
#       - neurons: 4 # layer 2, you can add however many layers you want. Number of neurons is changeable.
#         activation: relu # Don't forget to mention activation. First layer naturally has no activation; if you add it, it'll be ignored.
#       - neurons: 2 # layer 3
#         activation: softmax
# """)