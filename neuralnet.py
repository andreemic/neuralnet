
import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

learning_rate = 0.1


class Neuron:
    def __init__(self):
        self.output = 0
    def activate(self, inputs: np.ndarray, weights: np.ndarray):
        logit = np.dot(inputs, weights)
        self.output = sigmoid(logit)
        return self.output

class ConstantNeuron(Neuron):
    def __init__(self, val):
        super(Neuron, self).__init__()
        self.output = val
    def activate(self, inputs: np.ndarray, weights: np.ndarray):
        return self.output
    def set(self, val):
        self.output = val


class Layer:
    def __init__(self, size):
        self.neurons = [Neuron() for i in range(size)]
        self.y = np.zeros(size)
        self.size = size

    def propagate(self, inputs: np.ndarray, weights: np.ndarray):
        # inputs: vector of values (x)
        # weights: matrix where weights[i][j] is the weight going into i-th neuron from j-th input
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            # compute output for each neuron
            self.y[i] = neuron.activate(inputs, weights[i])
        return self.y


class Network:
    def __init__(self, input_size, output_size):
        # weights is an array of weight matricies
        self.weights = []
        self.input_size = input_size
        self.output_size = output_size

        self.input_layer = Layer(input_size)
        self.output_layer = Layer(output_size)
        self.layers = [self.input_layer, self.output_layer]

    def add_layer(self, layer: Layer):
        self.layers = self.layers[:-1] + [layer] + [self.layers[-1]]

    def init_weights(self):
        # each input goes straight to its dedicated neuron in the input layer
        self.weights = [np.identity(self.input_size)]

        for i in range(len(self.layers) - 1):
            this_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            self.weights.append(np.random.rand(next_layer.size, this_layer.size))

    def propagate(self, inputs: np.ndarray):
        y = inputs
        self.layers[0].y = inputs
        i = 1
        while i < len(self.layers):
            current_layer = self.layers[i]
            logits = np.dot(self.weights[i], y.reshape(len(y), 1))
            y = np.array([sigmoid(logit) for logit in logits])

            current_layer.y = y
            i += 1
        return y

    def train(self, training_examples):
        d_weights = self.backprop(training_examples)
        for i in range(len(self.weights)):
            self.weights[i] -= d_weights[i]

    # calculate derivatives of the SME error function
    #      E = 1/2 * sum(truth - prediction)^2
    # in respect to the activation of each neuron
    # returns list of vectors ([k][i] - derivative of activation of i-th neuron in k-th layer)
    def activation_derivatives(self, t):

        # calculate dE/da for all neuron outputs
        error_derivatives = []

        # pre-fill derivative table
        for layer in self.layers:
            error_derivatives.append(np.zeros(layer.size))

        # calculate output layer derivatives
        output_error_derivatives = -(t - self.output_layer.y)
        error_derivatives[-1] = output_error_derivatives

        # calculate hidden layer derivatives
        for i, layer in reversed(list(enumerate(self.layers))[:-1]):
            next_layer = self.layers[i + 1]
            next_layer_derivatives = error_derivatives[i + 1]

            next_layer_term = next_layer.y * (1 - next_layer.y) * next_layer_derivatives  # dE/dz
            weight_matrix = self.weights[i + 1]

            layer_derivatives = np.dot(next_layer_term, weight_matrix)

            error_derivatives[i] = layer_derivatives
        return error_derivatives

    def backprop(self, examples):

        # list of matrices to be added to each layer's weight matrix
        d_weights = [np.zeros(w.shape) for w in self.weights]

        # go over each example and add up the changes to be made to the weights between each layer
        for example in examples:
            x, t = example
            y = self.propagate(x)

            derivs = self.activation_derivatives(t)

            for i, layer in enumerate(self.layers[:-1]):
                next_layer = self.layers[i + 1]

                next_layer_term = (next_layer.y * (1 - next_layer.y) * derivs[i + 1])

                layer_d_weights = learning_rate * transpose_mul_vectors(next_layer_term, layer.y)

                d_weights[i + 1] += layer_d_weights

        return d_weights

    def example_error(self, example):
        x, t = example
        y = self.propagate(x)
        return np.linalg.norm(t - y) ** 2

    def network_error(self, examples):
        avg = 0
        for example in examples:
            avg += self.example_error(example)
        return avg / (2 * len(examples))


# couldn't figure out how to cleanly multiply vectors as matrices...
def transpose_mul_vectors(v1, v2):
    return np.dot(v1.reshape(len(v1), 1), v2.reshape(1, len(v2)))



import random
training_data = []
for i in range(10000):
    x = random.uniform(0, .5)
    y = random.uniform(0, .5)
    training_data.append((np.array([x, y]), np.array([x+y])))


mynet = Network(2, 1)
# mynet.add_layer(Layer(1))
mynet.init_weights()
print(training_data[:5])

print(mynet.network_error(training_data))
print(mynet.weights)

mynet.train(training_data)

print(mynet.network_error(training_data))
print(mynet.weights)

