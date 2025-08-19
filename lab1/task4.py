from random import uniform
from .task1 import neuron
import json

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
                              weight_min_value=-1, weight_max_value=1,
                              biases=[], activation_function='linear'):
        weights = [
            [uniform(weight_max_value, weight_min_value) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        return Layer(weights, biases, activation_function)


class NeuralNetwork:
    def __init__(self):
        self.layers = []


    def add_layer(self, weights, biases=[], activation_function='linear'):
        if len(self.layers) > 0:
            input_size = self.layers[-1].output_size
            if input_size != len(weights[0]):
                raise Exception('New layer has incompatible input size')

        new_layer = Layer(weights, biases, activation_function)
        self.layers.append(new_layer)

    
    def predict(self, input):
        for layer in self.layers:
            input = layer.calculate(input)
        return input


    @staticmethod
    def load_from_json_file(file_name):
        data = None
        with open(file_name, 'r') as f:
            data = json.load(f)
        network_weights = data['weights']
        network_biases = data['biases']
        network_activation_functions = data['activation_functions']
        nn = NeuralNetwork()
        for weights, biases, activation_function in zip(network_weights, network_biases, network_activation_functions):
            nn.add_layer(weights, biases, activation_function)
        return nn


if __name__ == '__main__':
    input = [0.5, 0.75, 0.1]
    weights1 = [[0.1, 0.1, -0.3],
                [0.1, 0.2, 0.0],
                [0.0, 0.7, 0.1],
                [0.2, 0.4, 0.0],
                [-0.3, 0.5, 0.1]]
    weights2 = [[0.7, 0.9, -0.4, 0.8, 0.1],
                [0.8, 0.5, 0.3, 0.1, 0.0],
                [-0.3, 0.9, 0.3, 0.1, -0.2]]
    nn = NeuralNetwork()
    nn.add_layer(weights1)
    nn.add_layer(weights2)

    # or load from a json file
    # nn = NeuralNetwork.load_from_file('test.json')

    print(nn.predict(input))
