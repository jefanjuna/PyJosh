refactor the current code into a class

convert dict to ordered list since above python 3.7, preserves insertion in dict then use indexing and apply methods to the respective layers instead of always doing f'layer{i}'

to add backpropagation, need to store activation values of each layer
then add backpropagation


btw all methods in fnn class is private because no public use cases yet

also pls dont push this network to the limits, for very deep networks might wanna consider batch normalization, residual connections and gradient clipping as a last resort for vanishing/exploding gradients problem

relu with he normal initialization is ald working hard enough to minimize such a problem but pls do remember to consider such exceptions for very deep neural networks

ok addressing the dying relu problem here, again he normal initialization and bath normalization plays a key part in minimizing this problem but again you can try changing the activation function to leaky relu instead. Also a tip, use lower learning rates and use residual connections for much deeper networks.

btw relu most common because easiest to compute but might wanna dig into more advanced options like leaky relu, prelu, elu, selu, gelu. All has their own slight differences of computation, benefits and disadvantages

now talking about backprop, relu is non differentiable at x = 0 so usually peopls just set it to 0 or 1 but 0 is most common because its emperically better and 1 had not much real benefits only use 1 for research purposes but do note that when it is set to 0 you may eencounter the dying relu problem






2nd session
using csv files for dataset
using pandas instead of csv library for importing csv datasets because its easier, more readable and can use chunkszie if needed to limit memory. It still uses more memory but is a good compromise and is slightly more preferred to handle edge cases in csv files. It also has seamless conversion to numpy arrays.


# Todo class_test reference
btw in the training loop there's probably some line like this:
activation = np.array(features_4[0])
some notes about that:
activation = features_4[0]
is the same as
activation = np.array(features_4[0])
but np.array creates a new array object in the memory, which is independent of the one in the list. This minimises accidental change to the original data but uses more memory as a result.
luckily, in this case, the data in activation is constantly going to be overwritten in the training process later on, so there will be no accumulation of memory usage.
although, if you have large activation vectors, creating copies may be resource intensive.
lastly, the frequency of this operation matter in the training loop, if copying multiple times each loop, it will lead to accumulation of memory usage.
by accidental change, I mean if you have a mutqable object in a mutable object and you reference it like so:
a = []
a.append(np.array([1, 2]))
b = []
b.append(a[0])
b[0][0] = 999
print(a[0])


btwwww during training, remember to lookout for exploding/vanishing gradients and dying relu problem.



config json file ntoes:
define neurons and layers
define activation functions
define loss functions
and maybe initialization methods


also newconfig file created for a new goal


another thing, my activation data input into the class needs more flexibility, I could code it into the class for flexibility but technically shouldn't this be all up to the user? It's up to them how they want the data to be parsed, reshaped and stuff.



ok now, do backprop first then scaling for more activation/loss functions.


I also wanna multithread the data loading process, load the next batch while the model is training the prvious on a separate thread than the training one so the training dosent lag behind.

cuda here would automatically does multithreading during training


update: use yaml instead of json for neural network config. although toml can be used more for settings config not for describing architecture. time to rewrite some code sigh :P

okay using yaml for config now, gonna add comments soon
