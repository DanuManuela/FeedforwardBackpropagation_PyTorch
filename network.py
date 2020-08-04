import torch
import random

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [torch.randn(y, 1) for y in sizes[1:]]
        self.weights = [torch.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
    def SGD(self, training_data, epochs, eta, test_data=None):
        for j in range(epochs):
            random.shuffle(training_data)
            for x, y in training_data:
                delta_grad_b, delta_grad_w = self.backprop(x, y)
                self.weights = [w - eta * nw for w, nw in zip(self.weights, delta_grad_w)]
                self.biases = [b - eta * nb for b, nb in zip(self.biases, delta_grad_b)]
            print('Epoch ', j, ' ', self.evaluate(test_data) / len(test_data) * 100, '%')
    def backprop(self, x, y):
        """Return a tuple ``(grad_b, grad_w)`` representing the gradient for the cost function C_x."""
        grad_b = [torch.zeros(b.shape) for b in self.biases]
        grad_w = [torch.zeros(w.shape) for w in self.weights]
        # feedforward
        activations = [x]  # list to store all the activations, layer by layer
        z = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z.append(torch.mm(w, activations[-1]) + b)
            activations.append(sigmoid(z[-1]))
        # backward pass
        delta = (activations[-1] - y) * sigmoid_prime(z[-1]) # BP1
        grad_b[-1] = delta # BP3
        grad_w[-1] = torch.mm(delta, torch.transpose(activations[-2], 0, 1)) # BP4
        for l in range(2, self.num_layers):
            delta = torch.mm(torch.transpose(self.weights[-l+1], 0, 1), delta) * sigmoid_prime(z[-l]) # BP2
            grad_b[-l] = delta # BP3
            grad_w[-l] = torch.mm(delta, torch.transpose(activations[-l - 1], 0, 1)) # BP4
        return grad_b, grad_w
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(torch.mm(w, a) + b)
        return a
    def evaluate(self, test_data):
        """Number of test inputs for which the neural network outputs the correct result"""
        test_results = [(torch.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

# Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))