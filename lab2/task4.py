import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from random import uniform
from .task3 import read_colors_dataset

model = Sequential()
weights = [np.array([[uniform(-1, 1) for _ in range(3)] for _ in range(4)])]
model.add(Dense(4, weights=weights, use_bias=False))
optimizer = SGD(learning_rate=0.01)
model.compile(optimizer, loss='mse')

training_inputs, training_expected_outputs = read_colors_dataset('datasets/colors/training_colors.txt')
for _ in range(20):
    for input, expected_output in zip(training_inputs, training_expected_outputs):
        model.fit(np.array([input]), np.array([expected_output]))

test_inputs, test_expected_outputs = read_colors_dataset('datasets/colors/test_colors.txt')
total_count = len(test_inputs)
correct_count = 0
for input, expected_output in zip(test_inputs, test_expected_outputs):
    output = model.predict(np.array([input]))[0]
    max_index = 0
    for i, value in enumerate(output):
        if value > output[max_index]:
            max_index = i
    if expected_output[max_index] == 1:
        correct_count += 1
print(f'correct={round(100 * correct_count / total_count, 2)}%')