from .task1 import neuron

def neural_network(input, layer_weights, biases=None, activation_function='linear'):
    neuron_count = len(layer_weights)
    if biases is None:
        biases = [0] * neuron_count
    output = []
    for weights, bias in zip(layer_weights, biases):
        output.append(neuron(input, weights, bias, activation_function))
    return output


if __name__ == '__main__':
    input = [0.5, 0.75, 0.1]
    layer_weights = [
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]
    ]
    
    print(neural_network(input, layer_weights))
