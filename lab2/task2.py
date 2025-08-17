from .task1 import train, layer_error
from lab1.task2 import neural_network

if __name__ == '__main__':
    inputs = [
        [0.5, 0.75, 0.1],
        [0.1, 0.3, 0.7],
        [0.2, 0.1, 0.6],
        [0.8, 0.9, 0.2]
    ]
    
    expected_outputs = [
        [0.1, 1.0, 0.1, 0.0, -0.1],
        [0.5, 0.2, -0.5, 0.3, 0.7],
        [0.1, 0.3, 0.2, 0.9, 0.1],
        [0.7, 0.6, 0.2, -0.1, 0.8]
    ]
    
    learning_rate = 0.01
    
    layer_weights = [
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.2, 0.4, 0.0],
        [-0.3, 0.5, 0.1]
    ]
    neuron_count = len(layer_weights)
    
    for i in range(1000):
        print(f'Epoch {i + 1}')
        total_error = 0
        for j, input in enumerate(inputs):
            expected_output = expected_outputs[j]
            output = neural_network(input, layer_weights)
            layer_weights = train(input, layer_weights, output, expected_output, learning_rate)
            error = layer_error(neuron_count, output, expected_output)
            total_error += error
            print(f'Series #{j + 1}')
            print(layer_weights)
            print(f'Error={error} ({error * neuron_count})')
            print()
        print(f'Error={total_error} ({total_error * neuron_count})\n')