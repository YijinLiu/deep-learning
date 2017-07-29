import argparse
import logging
import random

import numpy as np

import mnist

np.seterr(over="ignore")

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1).astype(np.float32) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x).astype(np.float32)
                for (y, x) in zip(sizes[1:], sizes[:-1])]

    def feed_forward(self, a):
        for (w, b) in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, training_samples, epochs, mini_batch_size,
            learning_rate, test_data = None):
        n = min(len(training_data), training_samples)
        for epoch in xrange(epochs):
            random.shuffle(training_data)
            for k in xrange(0, n, mini_batch_size):
                self.update_mini_batch(training_data[k:k+mini_batch_size], learning_rate)
            if test_data:
                correct = self.evaluate(test_data)
                logging.info("Epoch %d: %.2f(%d/%d)" % (epoch, float(correct) / len(test_data), correct,
                        len(test_data)))
            else:
                logging.info("Epoch %d is done." % epoch)

    def update_mini_batch(self, mini_batch, learning_rate):
        delta_b = [np.zeros(b.shape, dtype=np.float32) for b in self.biases]
        delta_w = [np.zeros(w.shape, dtype=np.float32) for w in self.weights]
        for x, y in mini_batch:
            tmp_delta_b, tmp_delta_w = self.back_propagate(x, y)
            delta_b = [a + b for a, b in zip(delta_b, tmp_delta_b)]
            delta_w = [a + b for a, b in zip(delta_w, tmp_delta_w)]
        self.biases = [a - (learning_rate / len(mini_batch)) * b for a, b in
                zip(self.biases, delta_b)]
        self.weights = [a - (learning_rate / len(mini_batch)) * b for a, b in
                zip(self.weights, delta_w)]

    def back_propagate(self, x, y):
        delta_b = [np.zeros(b.shape, dtype=np.float32) for b in self.biases]
        delta_w = [np.zeros(w.shape, dtype=np.float32) for w in self.weights]
        zs = []
        activation = x
        activations = [activation]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (activation - y) * sigmoid_derivative(zs[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_derivative(z)
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (delta_b, delta_w)

    def evaluate(self, test_data):
        results = [(np.argmax(self.feed_forward(x)) == y) for (x, y) in test_data]
        return sum(int(result) for result in results)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s[%(levelname)s] %(message)s", datefmt='%Y/%m/%d-%H:%M:%S',
            level=logging.INFO)
    parser = argparse.ArgumentParser(description='Simple 1 layer network.')
    parser.add_argument("--neurons", type=int, default=30,
            help="Number of neurons in the hidden layer.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--mini_batch_size", type=int, default=10, help="Mini batch size")
    parser.add_argument("--learning_rate", type=float, default=3.0, help="Learning rate")
    parser.add_argument("--training_samples", type=int, default=30000, help="Number of training samples")
    args = parser.parse_args()
    net = Network([28 * 28, args.neurons, 10])
    training_data = mnist.load("train", expand=True)
    testing_data = test_data=mnist.load("t10k")
    net.stochastic_gradient_descent(training_data, args.training_samples, args.epochs,
            args.mini_batch_size, args.learning_rate, test_data=testing_data)
