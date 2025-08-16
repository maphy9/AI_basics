from .task2 import neural_network

def deep_neural_network(input, network_weights, network_biases=[], network_activation_functions=[]):
    layers_count = len(network_weights)
    if len(network_biases) < layers_count:
        network_biases += [None] * (layers_count - len(network_biases))
    if len(network_activation_functions) < layers_count:
        network_activation_functions += [lambda x: x] * (layers_count - len(network_activation_functions))
    for layer_weights, biases, activation_function in zip(network_weights, network_biases, network_activation_functions):
        input = neural_network(input, layer_weights, biases, activation_function)
    return input


if __name__ == '__main__':
    input = [0.5, 0.75, 0.1]
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
    
    print(deep_neural_network(input, network_weights))
