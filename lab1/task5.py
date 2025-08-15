import numpy as np

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
weights1 = [np.array([
    [0.1, 0.1, 0, 0.2, -0.3],
    [0.1, 0.2, 0.7, 0.4, 0.5],
    [-0.3, 0, 0.1, 0, 0.1],
])]
weights2 = [np.array([
    [0.7, 0.8, -0.3],
    [0.9, 0.5, 0.9],
    [-0.4, 0.3, 0.3],
    [0.8, 0.1, 0.1],
    [0.1, 0, -0.2]
])]

model.add(Dense(5, weights=weights1, use_bias=False))
model.add(Dense(3, weights=weights2, use_bias=False))

input = np.array([[0.5, 0.75, 0.1]])

print(model.predict(input))
