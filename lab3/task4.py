from lab2.task3 import read_colors_dataset
from .task3 import NeuralNetwork, Layer

if __name__ == '__main__':
    train_inputs, train_expected_outputs = read_colors_dataset('datasets/colors/train_colors.txt')
    print('Read train dataset')

    # nn = NeuralNetwork()
    # nn.add_layer(Layer.generate_random_layer(3, 5, weight_min_value=0., activation_function='relu'))
    # nn.add_layer(Layer.generate_random_layer(5, 4))
    nn = NeuralNetwork.load_from_json('lab3/colors_network.json')
    learning_rate = 0.01

    iterations = 20
    progress = 0
    total_count = len(train_inputs) * iterations
    for _ in range(iterations):
        for input, expected_output in zip(train_inputs, train_expected_outputs):
            nn.fit(input, expected_output, learning_rate)
            print(' ' * 24, end='\r')
            progress += 1
            print(f'Progress: {round(100 * progress / total_count, 2)}%', end='')
    print(' ' * 24, end='\r')
    print('Training finished')

    del train_inputs
    del train_expected_outputs
    
    test_inputs, test_expected_outputs = read_colors_dataset('datasets/colors/test_colors.txt')
    print('Read test dataset')
    progress = 0
    total_count = len(test_inputs)
    correct_count = 0
    for input, expected_output in zip(test_inputs, test_expected_outputs):
        output = nn.predict(input) 
        max_index = 0
        for i, value in enumerate(output):
            if value > output[max_index]:
                max_index = i
        if expected_output[max_index] == 1:
            correct_count += 1
        progress += 1
        print(' ' * 50, end='\r')
        print(f'Progress: {round(100 * progress / total_count, 2)}%', end='; ')
        print(f'Correct percentage: {round(100 * correct_count / total_count, 2)}%', end='\r')
    print(' ' * 50, end='\r')
    print(f'Correct percentage: {round(100 * correct_count / total_count, 2)}%\n')

    nn.save_to_json('lab3/colors_network.json')
