from .task1 import train, layer_error
from lab1.task2 import neural_network
from random import uniform

def read_colors_dataset(file_name):
    inputs = []
    expected_outputs = []

    with open(file_name, 'r') as f:
        for line in f:
            data = line.split(' ')
            input1 = float(data[0])
            input2 = float(data[1])
            input3 = float(data[2])
            inputs.append([input1, input2, input3])

            output_color = int(data[3])
            expected_output = [0] * 4
            expected_output[output_color - 1] = 1
            expected_outputs.append(expected_output)

    return (inputs, expected_outputs)


if __name__ == '__main__':
    layer_weights = [[uniform(-1, 1) for _ in range(3)] for _ in range(4)]
    neuron_count = len(layer_weights)
    learning_rate = 0.01
    
    train_inputs, train_expected_outputs = read_colors_dataset('datasets/colors/train_colors.txt')
    iterations = 20
    total_count = len(train_inputs) * iterations
    progress = 0
    for iteration in range(iterations):
        for input, expected_output in zip(train_inputs, train_expected_outputs):
            output = neural_network(input, layer_weights)
            error = layer_error(neuron_count, output, expected_output)
            layer_weights = train(input, layer_weights, output, expected_output, learning_rate)
            print(' ' * 24, end='')
            progress += 1
            print(f'\rProgress: {round(100 * progress / total_count, 2)}%; Error={round(error, 5)}', end='')
    print('\nTraining finished\n')
    
    test_inputs, test_expected_outputs = read_colors_dataset('datasets/colors/test_colors.txt')
    total_count = len(test_inputs)
    progress = 0
    correct_count = 0
    for input, expected_output in zip(test_inputs, test_expected_outputs):
        output = neural_network(input, layer_weights)
        max_index = 0
        for i, value in enumerate(output):
            if value > output[max_index]:
                max_index = i
        if expected_output[max_index] == 1:
            correct_count += 1
        progress += 1
        print(' ' * 24, end='')
        print(f'\rProgress: {round(100 * progress / total_count, 2)}%; correct={round(100 * correct_count / total_count, 2)}%', end='')
    print(f'\ncorrect={round(100 * correct_count / total_count, 2)}%\n')
    
