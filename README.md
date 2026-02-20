# PyJosh

PyJosh is a lightweight neural network framework built from scratch using **JAX**, inspired by the design philosophy and modular architecture of PyTorch — hence the name.

The goal was to deeply understand how modern deep learning frameworks work internally — including forward propagation, gradient computation, parameter updates, and extensibility.
Rather than using high-level libraries, this project implements the core training pipeline manually while keeping the design clean and extensible.

Since this project is just for learning, it’s not meant to be production ready. However, it’s designed to be:

* Easy to modify
* Easy to extend
* Easy to experiment with

You can add new activation functions, loss functions, or change the architecture without changing much outside the class.

---

## What it does

* Feedforward fully connected neural network
* Configurable architecture via `config.yml`
* ReLU + Softmax (easy to extend)
* Cross-entropy loss (easy to add more)
* Gradient-based training using `jax.value_and_grad`
* Full control over dataset loading

---

## Main project structure

```
neural_network_class.py   # Core framework
class_use.py              # Framework usage
config.yml                # Network structure
data.csv                  # Example dataset
```

---

## Configuration example

`config.yml`

```yaml
layers:
- neurons: 3
- neurons: 4
  activation: relu
- neurons: 2
  activation: softmax
```

You can define any number of layers.

To support a new activation:

1. Add the function inside the class
2. Register it in `_apply_activation`
3. Use it in `config.yml`

Same idea for new loss functions.

---

## Dataset example

The example dataset (`data.csv`) looks like this:

```
Features,Ground Truth
"1,1,1","0.1,0.9"
"2,2,2","0.9,0.1"
```

But you're not limited to this format.

You can:

* Load data however you want
* Convert images into numeric arrays
* Create your own preprocessing pipeline

As long as you pass:

```
x -> (batch_size, input_dim)
y -> (batch_size, output_dim)
```

the network will work.

---

## To run example

First, install the required dependencies:
```
pip install jax jaxlib pandas pyyaml
```

Then run:
```
python class_use.py
```
