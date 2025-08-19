from utils.activation_functions import activation_functions

def neuron(input, weights, bias=0, activation_function='linear'):
    if len(input) != len(weights):
        raise Exception('Neuron input vector and weights vector have different sizes')
    output = 0
    for x, w in zip(input, weights):
        output += x * w
    f, _ = activation_functions[activation_function]
    return f(output + bias)

if __name__ == '__main__':
    input = [0.5, 0.75, 0.1]
    layer_weights = [
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]
    ]

    output = []
    for weights in layer_weights:
        output.append(neuron(input, weights))
    print(output)
