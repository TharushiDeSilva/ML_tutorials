# This file tries to improve the answer given by simple_neural_network.py by increasing the data set size

from statistics import mode
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(units = 1, input_shape = [1])])
model.compile(optimizer='sgd', loss='mean_squared_error') 

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0], dtype=float)

model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))

 
#original answer = 18.977282 (error - 0.022718)

# increase the data set size from 6 to 20 
#answer - 19.006958 (error - 0.006958)
# prediction is improved. but cannot increse the data set size without making a huge anomaly in the result 
 