from lab1.task2 import neural_network
from lab1.task3 import deep_neural_network
from utils.activation_functions import activation_functions
from utils.math import *

def train(input, network_weights, expected_output, learning_rate=0.1, network_biases=[], network_activation_functions=[]):
    layers_count = len(network_weights)
    if len(network_biases) < layers_count:
        network_biases += [None] * (layers_count - len(network_biases))
    if len(network_activation_functions) < layers_count:
        network_activation_functions += ['linear'] * (layers_count - len(network_activation_functions))

    inputs = [input]
    for layer_weights, biases, activation_function in zip(network_weights, network_biases, network_activation_functions):
        input = neural_network(input, layer_weights, biases, activation_function)
        inputs.append(input)

    neuron_count = len(network_weights[-1])
    output_layer_delta = subtract_vectors(inputs[-1], expected_output)
    output_layer_delta = multiply_vector_by_scalar(output_layer_delta, 2 / neuron_count)
    layer_deltas = [output_layer_delta]
    for i in range(layers_count - 2, -1, -1):
        next_layer_delta = layer_deltas[-1]
        layer_weights_transposed = transpose_matrix(network_weights[i + 1])
        layer_delta = neural_network(next_layer_delta, layer_weights_transposed)
        layer_deltas.append(layer_delta)
    layer_deltas = layer_deltas[::-1]
    
    for i in range(layers_count):
        output = inputs[i + 1]
        activation_function = network_activation_functions[i]
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
        network_weights[i] = subtract_matrices(network_weights[i], weights_delta)


if __name__ == '__main__':
    inputs = [
        [0.5, 0.75, 0.1],
        [0.1, 0.3, 0.7],
        [0.2, 0.1, 0.6],
        [0.8, 0.9, 0.2]
    ]

    network_weights = [
        [
            [0.1, 0.1, -0.3],
            [0.1, 0.2, 0.0],
            [0.0, 0.7, 0.1],
            [0.2, 0.4, 0.0],
            [-0.3, 0.5, 0.1]
        ],
        [
            [0.7, 0.9, -0.4, 0.8, 0.1],
            [0.8, 0.5, 0.3, 0.1, 0.0],
            [-0.3, 0.9, 0.3, 0.1, -0.2]
        ]
    ]

    expected_outputs = [
        [0.1, 1.0, 0.1],
        [0.5, 0.2, -0.5],
        [0.1, 0.3, 0.2],
        [0.7, 0.6, 0.2]
    ]

    learning_rate = 0.01
    network_activation_functions = ['relu']

    for i in range(50):
        print(f'Epoch {i + 1}')
        for input, expected_output in zip(inputs, expected_outputs):
            output = deep_neural_network(input, network_weights)
            print(output)
            train(input, network_weights, expected_output, learning_rate, [], network_activation_functions)
        print('')

