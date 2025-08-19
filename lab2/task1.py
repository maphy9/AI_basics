from lab1.task2 import neural_network
from utils.math import *

def layer_error(neuron_count, output, expected_output):
    error = 0
    for o, eo in zip(output, expected_output):
        error += (o - eo) ** 2
    return error / neuron_count


def train(input, layer_weights, output, expected_output, learning_rate=0.1):
    neuron_count = len(layer_weights)
    delta = subtract_vectors(output, expected_output)
    delta = multiply_vector_by_scalar(delta, learning_rate * 2 / neuron_count)
    delta = vector_outer_product(delta, input)
    return subtract_matrices(layer_weights, delta)


if __name__ == '__main__':
    layer_weights = [[0.5]]
    neuron_count = len(layer_weights)
    expected_output = [0.8]
    input = [2]
    learning_rate = 0.1
    
    for i in range(20):
        print(f'Epoch {i + 1}')
        print(layer_weights)
        output = neural_network(input, layer_weights)
        error = layer_error(neuron_count, output, expected_output)
        print(f'Output={output}')
        print(f'Error={error}\n')
        layer_weights = train(input, layer_weights, output, expected_output, learning_rate)
