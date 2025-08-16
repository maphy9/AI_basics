from .task1 import train, layer_error
from random import uniform

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
    for i in range(10):
        print(f'Epoch {i + 1}')
        print(layer_weights)
        total_error = 0
        for input, expected_output in zip(inputs, expected_outputs):
            _, error = layer_error(input, layer_weights, expected_output)
            total_error += error
        print(f'Error={total_error}\n')
        for input, expected_output in zip(inputs, expected_outputs):
            layer_weights = train(input, layer_weights, expected_output, learning_rate)