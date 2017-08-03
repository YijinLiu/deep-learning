import logging
import random

import numpy as np

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
        delta = activation
        delta[y] -= 1.0
        for l in xrange(1, self.num_layers):
            z = zs[-l]
            delta = delta * sigmoid_derivative(zs[-l])
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
            delta = np.dot(self.weights[-l].transpose(), delta)
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
    import argparse

    import mnist

    import img_util

    logging.basicConfig(format="%(asctime)s[%(levelname)s] %(message)s", datefmt='%Y/%m/%d-%H:%M:%S',
            level=logging.INFO)
    parser = argparse.ArgumentParser(description='Simple 1 layer network.')
    parser.add_argument("--neurons", type=int, default=30,
            help="number of neurons in the hidden layer")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--mini_batch_size", type=int, default=10, help="mini batch size")
    parser.add_argument("--learning_rate", type=float, default=3.0, help="learning rate")
    parser.add_argument("--training_samples", type=int, default=30000,
            help="number of training samples to use per epoch")
    parser.add_argument("--down_sample_rate", type=int, default=1, help="down sample the images")
    args = parser.parse_args()
    size = mnist.IMAGE_SIZE / args.down_sample_rate
    net = Network([size * size, args.neurons, 10])
    training_data = mnist.load("train")
    testing_data = test_data=mnist.load("t10k")
    if args.down_sample_rate > 1:
        def down_sample(sample):
            return (img_util.down_sample(sample[0], mnist.IMAGE_SIZE, mnist.IMAGE_SIZE,
                                         args.down_sample_rate, args.down_sample_rate),
                    sample[1])
        training_data = map(down_sample, training_data)
        testing_data = map(down_sample, testing_data)

    net.stochastic_gradient_descent(training_data, args.training_samples, args.epochs,
            args.mini_batch_size, args.learning_rate, test_data=testing_data)

'''
MKL_NUM_THREADS=1 python simple.py --training_samples=10000 --down_sample_rate=2
Epoch 0: 0.86(8604/10000)
Epoch 1: 0.90(9048/10000)
Epoch 2: 0.91(9109/10000)
Epoch 3: 0.91(9084/10000)
Epoch 4: 0.92(9188/10000)
Epoch 5: 0.93(9256/10000)
Epoch 6: 0.93(9299/10000)
Epoch 7: 0.93(9292/10000)
Epoch 8: 0.93(9328/10000)
Epoch 9: 0.93(9323/10000)
Epoch 10: 0.94(9353/10000)
Epoch 11: 0.94(9389/10000)
Epoch 12: 0.93(9325/10000)
Epoch 13: 0.94(9377/10000)
Epoch 14: 0.94(9374/10000)
Epoch 15: 0.94(9434/10000)
Epoch 16: 0.94(9385/10000)
Epoch 17: 0.95(9474/10000)
Epoch 18: 0.94(9413/10000)
Epoch 19: 0.94(9449/10000)
Epoch 20: 0.95(9458/10000)
Epoch 21: 0.94(9446/10000)
Epoch 22: 0.95(9479/10000)
Epoch 23: 0.95(9472/10000)
Epoch 24: 0.95(9508/10000)
Epoch 25: 0.95(9509/10000)
Epoch 26: 0.95(9501/10000)
Epoch 27: 0.95(9493/10000)
Epoch 28: 0.95(9520/10000)
Epoch 29: 0.95(9497/10000)
'''
