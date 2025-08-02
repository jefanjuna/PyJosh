layers:
- neurons: 3
  activation: relu
  dropout: 0.2
- neurons: 4
  activation: relu
  dropout: 0.2
- neurons: 2
  activation: softmax
  dropout: 0

data (storage paths)
hyperparameters (training) (optimizer? learning rate, batch size, epochs, loss function, momentum?)
logging?
model checkpointing


may not even need "name" btw