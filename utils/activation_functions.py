from math import exp

def linear(x):
    return x


def linear_derivative(_):
    return 1


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return 1 if x > 0 else 0


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


activation_functions = {
    'linear': (linear, linear_derivative),
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative)
}
