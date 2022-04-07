# This file tries to check the accuracy of the answer given by simple_neural_network.py by introducing a small anomay into the dataset

from statistics import mode
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(units = 1, input_shape = [1])])
model.compile(optimizer='sgd', loss='mean_squared_error') 

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.89, 13.0], dtype=float)

model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))
print(model)

# original answer = 18.977282 (error - 0.022718)

# add an anomay in the results set
#answer 18.928946  (error - much larger)

 