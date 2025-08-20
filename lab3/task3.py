from random import uniform
from lab1.task1 import neuron
import json
from lab1.task2 import neural_network
from utils.activation_functions import activation_functions
from utils.math import *

class Layer:
    def __init__(self, weights, biases=[], activation_function='linear'):
        self.input_size = len(weights[0])
        self.output_size = len(weights)
        self.weights = weights
        self.biases = biases
        if len(biases) < self.output_size:
            self.biases += [0] * (self.output_size - len(biases))
        self.activation_function = activation_function


    def calculate(self, input):
        output = []
        for weights, bias in zip(self.weights, self.biases):
            output.append(neuron(input, weights, bias, self.activation_function))
        return output


    @staticmethod
    def generate_random_layer(input_size, output_size,
                              weight_min_value=-1., weight_max_value=1.,
                              biases=[], activation_function='linear'):
        weights = [
            [uniform(weight_max_value, weight_min_value) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        return Layer(weights, biases, activation_function)


class NeuralNetwork:
    def __init__(self):
        self.layers = []


    def add_layer(self, layer):
        if len(self.layers) > 0:
            input_size = self.layers[-1].output_size
            if input_size != layer.input_size:
                raise Exception('New layer has incompatible input size')
        self.layers.append(layer)

    
    def predict(self, input):
        for layer in self.layers:
            input = layer.calculate(input)
        return input


    def fit(self, input, expected_output, learning_rate=0.1):
        layers_count = len(self.layers)

        inputs = [input]
        for layer in self.layers:
            input = layer.calculate(input)
            inputs.append(input)

        output_layer_delta = subtract_vectors(inputs[-1], expected_output)
        output_layer_delta = multiply_vector_by_scalar(output_layer_delta, 2 / self.layers[-1].output_size)
        layer_deltas = [output_layer_delta]
        for i in range(layers_count - 2, -1, -1):
            next_layer_delta = layer_deltas[-1]
            layer_weights = self.layers[i + 1].weights
            layer_weights_transposed = transpose_matrix(layer_weights)
            layer_delta = neural_network(next_layer_delta, layer_weights_transposed)
            layer_deltas.append(layer_delta)
        layer_deltas = layer_deltas[::-1]
        
        for i in range(layers_count):
            output = inputs[i + 1]
            activation_function = self.layers[i].activation_function
            _, derivative = activation_functions[activation_function]
            output = apply_function_to_vector(output, derivative)
            layer_deltas[i] = vector_hadamard_product(layer_deltas[i], output)

        for i in range(layers_count):
            layer_delta = layer_deltas[i]
            input = inputs[i]
            if len(layer_delta) > len(input):
                input = multiply_vector_by_scalar(input, learning_rate)
            else:
                layer_delta = multiply_vector_by_scalar(layer_delta, learning_rate)
            weights_delta = vector_outer_product(layer_delta, input)
            self.layers[i].weights = subtract_matrices(self.layers[i].weights, weights_delta)


    def save_to_json(self, file_name):
        data = {
                'weights': [],
                'biases': [],
                'activation_functions': []
        }
        for layer in self.layers:
            weights = layer.weights
            biases = layer.biases
            activation_function = layer.activation_function
            data['weights'].append(weights)
            data['biases'].append(biases)
            data['activation_functions'].append(activation_function)
        with open(file_name, 'w') as f:
            json.dump(data, f)


    @staticmethod
    def load_from_json(file_name):
        data = None
        with open(file_name, 'r') as f:
            data = json.load(f)
        network_weights = data['weights']
        network_biases = data['biases']
        network_activation_functions = data['activation_functions']
        nn = NeuralNetwork()
        for weights, biases, activation_function in zip(network_weights, network_biases, network_activation_functions):
            layer = Layer(weights, biases, activation_function)
            nn.add_layer(layer)
        return nn


def read_labels(file_name):
    labels = None
    with open(file_name, 'rb') as f:
        magic_number = int.from_bytes(f.read(4))
        if magic_number != 2049:
            raise Exception('Labels file is corrupted')
        number_of_data = int.from_bytes(f.read(4))
        labels = list(f.read(number_of_data))
    return labels


def read_images(file_name):
    images = None
    with open(file_name, 'rb') as f:
        magic_number = int.from_bytes(f.read(4))
        if magic_number != 2051:
            raise Exception('Images file is corrupted')
        number_of_images = int.from_bytes(f.read(4))
        number_of_rows = int.from_bytes(f.read(4))
        number_of_cols = int.from_bytes(f.read(4))
        images = [[]] * number_of_images
        for i in range(number_of_images):
            images[i] = [pixel / 255 for pixel in list(f.read(number_of_rows * number_of_cols))]
    return images 


if __name__ == '__main__':
    train_labels = read_labels('datasets/MNIST/train-labels.idx1-ubyte')
    print('Read train labels')
    train_images = read_images('datasets/MNIST/train-images.idx3-ubyte')
    print('Read train images')

    # nn = NeuralNetwork()
    # nn.add_layer(Layer.generate_random_layer(784, 40, weight_min_value=-0.1, weight_max_value=0.1, activation_function='relu'))
    # nn.add_layer(Layer.generate_random_layer(40, 10, weight_min_value=-0.1, weight_max_value=0.1))
    nn = NeuralNetwork.load_from_json('lab3/neural_network.json')
    learning_rate = 0.01

    progress = 0
    total = len(train_images)
    iterations = 2
    for _ in range(iterations):
        for image, label in zip(train_images, train_labels):
            expected_output = [0] * 10
            expected_output[label] = 1
            nn.fit(image, expected_output, learning_rate)
            progress += 1
            print(' ' * 20, end='\r')
            print(f'Progress: {round(100 * progress / (total * iterations), 2)}%', end='\r')
    print(' ' * 20, end='\r')
    print('Training finished')

    del train_images
    del train_labels

    test_labels = read_labels('datasets/MNIST/t10k-labels.idx1-ubyte')
    print('Read test labels')
    test_images = read_images('datasets/MNIST/t10k-images.idx3-ubyte')
    print('Read test images')

    correct_count = 0
    total = len(test_images)
    for image, label in zip(test_images, test_labels):
        output = nn.predict(image)
        max_index = 0
        for i in range(len(output)):
            if output[i] > output[max_index]:
                max_index = i
        if label == max_index:
            correct_count += 1
        print(' ' * 35, end='\r')
        print(f'Correct percentage: {round(100 * correct_count / total, 2)}%', end='\r')
    print(' ' * 35, end='\r')
    print(f'Correct percentage: {round(100 * correct_count / total, 2)}%')

    nn.save_to_json('lab3/neural_network.json')
